#include "hbflab/sampled_readonly.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace hbflab {
namespace {
void need(bool value,const char *message){if(!value)throw std::invalid_argument(message);}
std::uint64_t padded(std::uint64_t bytes){
    return hbf_eventlab::checked_mul(hbf_eventlab::checked_add(bytes,SampledRegistration::alignment-1)/SampledRegistration::alignment,SampledRegistration::alignment);
}
}
SampledRoute sampled_route(const std::string &name){
    if(name=="symbolic")return SampledRoute::Symbolic;
    if(name=="ptx")return SampledRoute::Ptx;
    if(name=="cuda")return SampledRoute::Cuda;
    throw std::invalid_argument("unsupported sampled readonly route");
}
const char *sampled_route_name(SampledRoute route){
    switch(route){case SampledRoute::Symbolic:return "symbolic";case SampledRoute::Ptx:return "ptx";case SampledRoute::Cuda:return "cuda";}
    throw std::invalid_argument("invalid sampled readonly route enum");
}
SampledRegistration::SampledRegistration(SampledOptions options):options_(options){
    sampled_route_name(options.route);options.geometry.validate();
    need(options.geometry.host_channels==1 && options.geometry.dies_per_channel==1 &&
         options.geometry.banks_per_die==2 && options.geometry.pages_per_block==4,"sampled media geometry unsupported");
    need(options.layout_bytes && options.layout_bytes%alignment==0 && options.kv_bytes%alignment==0,"sampled layout alignment");
    need(hbf_eventlab::checked_add(options.layout_bytes,options.kv_bytes)<=options.geometry.channel_bytes(),"sampled media capacity insufficient");
}
void SampledRegistration::current() const{
    if(owner_!=std::this_thread::get_id() || closed_)throw std::logic_error("sampled owner or lifecycle mismatch");
}
void SampledRegistration::add(SampledRange range){
    current();need(!prepared_ && ranges_.size()<max_ranges,"sampled registration frozen or full");
    const auto address=reinterpret_cast<std::uintptr_t>(range.address);
    need(!range.name.empty() && address && range.bytes && range.bytes<=std::numeric_limits<std::uintptr_t>::max()-address,"sampled physical range invalid");
    const auto reserved=padded(range.bytes);
    need(range.logical%alignment==0 && range.logical<=options_.layout_bytes && reserved<=options_.layout_bytes-range.logical,"sampled logical range invalid");
    for(const auto &other:ranges_){
        const auto base=reinterpret_cast<std::uintptr_t>(other.address);
        need(range.name!=other.name,"sampled duplicate tensor name");
        need(!(address>=base?address-base<other.bytes:base-address<range.bytes),"sampled physical range overlap");
        need(!(range.logical>=other.logical?range.logical-other.logical<padded(other.bytes):other.logical-range.logical<reserved),"sampled logical range overlap");
    }
    registered_bytes_=hbf_eventlab::checked_add(registered_bytes_,range.bytes);
    initialized_bytes_=hbf_eventlab::checked_add(initialized_bytes_,reserved);
    ranges_.push_back(std::move(range));
}
void SampledRegistration::prepare(){
    current();need(!prepared_ && !ranges_.empty(),"sampled prepare invalid");
    std::sort(ranges_.begin(),ranges_.end(),[](const auto &a,const auto &b){return a.logical<b.logical;});
    AsyncRuntimeOptions runtime;runtime.backend.geometry=options_.geometry;
    runtime.backend.max_events_per_batch=2000000000ull;
    runtime.backend.resident_fragments=256;
    runtime.backend.coalesce_capacity_feedback=true;runtime.backend.direct_device_transactions=true;
    runtime.readonly_layout_bytes=options_.layout_bytes;runtime.readonly_reservation_alignment=alignment;
    runtime.audit_limit=8193;runtime.timeout_ns=1000000000ull;runtime.service_timeout_ns=60000000000ull;
    for(const auto &range:ranges_)runtime.readonly_bases.push_back(range.logical);
    service_=std::make_unique<LiveTimingService>(runtime,2048,1);
    for(const auto &range:ranges_)service_->register_readonly(range.address,range.bytes);
    need(service_->statistics().backend.device.programs==initialized_bytes_/4096,"sampled weight initialization count");
    prepared_=true;
}
void SampledRegistration::prepare_kv(const std::function<void(LiveTimingService *)> &initialize){
    current();need(prepared_ && !started_ && !kv_prepared_ && options_.kv_bytes,"sampled KV registration order");
    initialize(service_.get());
    need(service_->statistics().backend.device.programs==(initialized_bytes_+options_.kv_bytes)/4096,"sampled KV reservation budget mismatch");
    kv_prepared_=true;
}
void SampledRegistration::start(){
    current();need(prepared_ && !started_ && (!options_.kv_bytes || kv_prepared_),"sampled service start order");
    service_->start();started_=true;
}
LiveTimingService &SampledRegistration::service(){current();need(prepared_,"sampled service not prepared");return *service_;}
void SampledRegistration::close(){
    current();need(started_,"sampled close before start");service_->finish();
    const auto stats=service_->statistics();
    need(service_->drained() && stats.submitted==stats.completed && stats.media_submitted==stats.media_completed &&
         !stats.audit_dropped && stats.backend.peak_resident_fragments<=256 &&
         stats.backend.registered_globals==stats.backend.completed_globals && !stats.backend.host.logical_failed,"sampled shared service not drained");
    service_.reset();closed_=true;
}
}
