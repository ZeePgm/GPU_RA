"""Seal the independent build and report the completed correctness matrix."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,difflib,re
R=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
    task=read(R/'provenance/task.json')
    assert sha(R/'configs/prompt.txt')==task['prompt_sha256']
    assert sha(Path(task['model']))==task['model_sha256']
    build_text='\n'.join((R/p).read_text(encoding='utf-8') for p in ['build/model/build.ninja','build/model/CMakeFiles/rules.ninja'])
    forbidden=re.findall(r'E\$?:[/\\]+v1_update[/\\]+(?:build-[^\s]+|output[/\\]+[^\s]+)',build_text,re.I)
    assert not forbidden,forbidden
    write(R/'provenance/build-closure.json',{'status':'pass','old_workspace_build_or_output_dependencies':forbidden,'new_binary_directory':str(R/'build/model/bin')})
    batch=R/'results/correctness-001'
    accepted=[read(batch/f'{scale}-acceptance.json') for scale in ['F1','F2']]
    assert all(a['status']=='pass' for a in accepted)
    assert sha(batch/'F1-native/logits.f32')==sha(batch/'F2-native/logits.f32')
    historic=R.parents[1]/'output/readonly-f-compare-a-20261108/native-old'
    assert sha(batch/'F1-native/logits.f32')==sha(historic/'logits.f32')
    write(R/'provenance/historical-native-check.json',{'status':'pass','reference':str(historic),'logits_sha256':sha(historic/'logits.f32'),'new_native_sha256':sha(batch/'F1-native/logits.f32')})
    assert sha(batch/'F1-joint-modeled/logits.f32')==sha(batch/'F2-joint-modeled/logits.f32')
    assert read(batch/'F1-joint-modeled/kv-content.json')['sequences']==read(batch/'F2-joint-modeled/kv-content.json')['sequences']
    profile=read(R/'configs/qwen3-0.6b.json')
    expected_kv_pages=2*profile['layers']*((51+profile['block_tokens']-1)//profile['block_tokens'])*profile['pages_per_allocation']
    for scale in ['F1','F2']:
        for label in ['kv-bypass','kv-modeled','joint-bypass','joint-modeled','joint-host']:
            assert read(batch/f'{scale}-{label}'/'kv-content.json')['allocated_pages']==expected_kv_pages
    fixture=read(R/'results/b2-fixture.json');assert fixture['status']=='pass' and len(fixture['cases'])==28
    core=(R/'results/test-core.log').read_text(encoding='utf-8-sig');assert '100% tests passed, 0 tests failed' in core
    assert 'PASS media-only' in (R/'results/media-only.log').read_text()
    assert 'terminal status 1' in (R/'results/b2-negative.log').read_text()
    origins=read(R/'provenance/source-origin.json');changed=[];patch=[]
    for o in origins:
        p=R/'source'/o['path'];h=sha(p)
        if h!=o['sha256']:
            changed.append({'path':o['path'],'original_sha256':o['sha256'],'integrated_sha256':h})
            if p.suffix in ['.cpp','.cu','.hpp','.txt','.h','.cuh']:
                before=Path(o['source']).read_text(encoding='utf-8').splitlines(True)
                patch.extend(difflib.unified_diff(before,p.read_text(encoding='utf-8').splitlines(True),fromfile='original/'+o['path'],tofile='integrated/'+o['path']))
    (R/'provenance/integration.patch').write_text(''.join(patch),encoding='utf-8')
    write(R/'provenance/changed-files.json',changed)
    original_names={str(Path(o['path'])) for o in origins}
    new_files=[{'path':str(p.relative_to(R/'source')),'sha256':sha(p)} for p in sorted((R/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and str(p.relative_to(R/'source')) not in original_names]
    write(R/'provenance/new-files.json',new_files)
    manifest=[]
    for folder in ['source','scripts','configs']:
        for p in sorted((R/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:manifest.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)})
    for p in sorted((R/'build/model/bin').glob('*')):
        if p.is_file():manifest.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)})
    p=R/'README.md';manifest.append({'path':'README.md','bytes':p.stat().st_size,'sha256':sha(p)})
    write(R/'provenance/final-source-build-manifest.json',manifest)
    records=[read(p) for p in sorted(batch.glob('*.process.json'))];assert len(records)==16 and all(p['returncode']==0 and not p['error'] for p in records)
    module_checks=0
    for p in batch.glob('*.modules.json'):
        for row in read(p):
            assert sha(Path(row['path']))==row['sha256'],row['path']
            module_checks+=1
    assert module_checks==80
    ended=datetime.now(timezone.utc);start=datetime.fromisoformat(read(R/'provenance/task.json')['start_utc'].replace('Z','+00:00'))
    rows=[]
    for scale in ['F1','F2']:
        case=batch/f'{scale}-joint-modeled';j=read(case/'joint-summary.json');k=read(case/'kv-summary.json');a=next(x for x in accepted if x['scale']==scale)
        error=next(x for x in a['comparisons'] if x['case']==f'{scale}-joint-modeled')
        rows.append({'scale':scale,'requests':j['requests'],'covered_MiB':j['covered_bytes']//1024**2,'publications':j['publications'],'dependency_checks':j['dependency_checks'],'kv_media_requests':j['runtime']['media_completed'],'max_abs':error['max_abs_vs_native'],'rms':error['rms_vs_native'],'process_s':read(batch/f'{scale}-joint-modeled.process.json')['wall_s'],'host_process_s':read(batch/f'{scale}-joint-host.process.json')['wall_s']})
    result={'status':'pass','scope':read(R/'provenance/task.json')['scope'],'start_utc':start.isoformat(),'end_utc':ended.isoformat(),'elapsed_s':(ended-start).total_seconds(),'accepted_model_processes':16,'model_process_sum_s':sum(x['wall_s'] for x in records),'GPU_fixture_cases':28,'loaded_module_hash_checks':module_checks,'core_test_summary':re.findall(r'100% tests passed[^\n]*',core),'rows':rows,'cross_scale_logits_and_KV_exact':True,'statistical_performance_confirmation':False}
    write(R/'reports/acceptance.json',result)
    lines=['# 双线联合正确性结果','',f'状态：通过。Qwen3-0.6B F16，44 输入 / 8 输出，按 F1 完成验收后进入 F2。共 16 个完整模型进程通过；未做统计性能确认。','',
        '| 规模 | 权重请求 | 完整读取 MiB | 矩阵发布 | 逐 lane 依赖检查 | KV 介质请求 | native 最大绝对误差 / RMS | 联合进程秒 |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for x in rows:lines.append(f"| {x['scale']} | {x['requests']} | {x['covered_MiB']} | {x['publications']} | {x['dependency_checks']} | {x['kv_media_requests']} | {x['max_abs']:.6f} / {x['rms']:.6f} | {x['process_s']:.3f} |")
    lines+=['','全部输入/输出 token 一致；只读对 native 及同算术组间完整 logits 逐字节一致。纯 KV、联合路径、host/device 写来源的最终 KV 全内容与 44-token 前缀哈希一致。原 native 数值容差保持不变。','',
        '权重和 KV 共用一个服务；Host resident fragments 不超过 256。8 次事务完成后序列长度为 51（44-token prefill 加 7 次 decode），正常结束 allocation、pin、dirty、lease、pinned 均为零。初始化 program 仅包含选定权重与 KV，未选权重地址间隙没有被初始化。','',
        f"组件验证：{result['core_test_summary'][0]}；B2 28 个 GPU 边界/慢生产者/复用场景及非法范围负例通过；纯 KV 服务启动、读写读及排空通过。",'',
        f"实际总耗时：{result['elapsed_s']/60:.2f} 分钟，包含源码核验、实现、构建、组件验证、模型运行与结果复核；16 个模型进程累计 {result['model_process_sum_s']/60:.2f} 分钟。起止 UTC：{result['start_utc']} 至 {result['end_utc']}。",'',
        '运行秒数包含初始化、逐 lane 验证、全 KV 字节审计与日志，仅用于本次执行成本记录。共同配置和重现入口见目录 README；每进程参数、实际加载模块与哈希、日志及机器验收见 results/correctness-001。','',
        '首次沙箱构建的 MSVC C1902 日志保留在 results/configure-sandbox-failure.log；测试接线脚本修正了默认 GBK 解码，新建纯 KV 夹具补齐 CUDA 上下文初始化。对应接线/上下文失败日志保留，未通过放宽模型验收门槛解决。源码与二进制最终清单见 provenance/final-source-build-manifest.json；scripts/verify.py 可只读复核全部清单。']
    (R/'reports/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    evidence=[]
    for directory in ['results','provenance','reports']:
        for p in sorted((R/directory).rglob('*')):
            if p.is_file() and p.name!='evidence-manifest.json':evidence.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)})
    write(R/'provenance/evidence-manifest.json',evidence)
    for x in manifest+evidence:assert sha(R/x['path'])==x['sha256']
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
