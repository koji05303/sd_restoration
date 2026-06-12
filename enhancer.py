from sd_enhancer.cli import main
from sd_enhancer.config import EnhanceConfig


def enhance_image(config: EnhanceConfig, pipe=None):
    from sd_enhancer.pipeline import enhance_image as run_enhance_image

    return run_enhance_image(config, pipe=pipe)


if __name__ == "__main__":
    raise SystemExit(main())
