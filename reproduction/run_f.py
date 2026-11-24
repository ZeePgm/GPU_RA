"""Compatibility entry point for the FFN weight-read runner."""
if __package__:
    from .run_ffn_weight_reads import main
else:
    from run_ffn_weight_reads import main


if __name__ == '__main__':
    main()
