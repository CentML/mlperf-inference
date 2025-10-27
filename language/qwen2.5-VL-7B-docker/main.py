import mlperf_loadgen as lg
import os
import logging
import sys

from enum import Enum
from typing import Annotated, Optional, Literal

from typing_extensions import Annotated
from pydantic import Field, BaseModel
import typer
import pydantic_typer

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Qwen2-5-7B-MAIN")


class Scenario(str, Enum):
    Offline = "Offline"
    Server = "Server"
    SingleStream = "SingleStream"


DType = Literal["float16", "bfloat16", "float32"]


# ---- Pydantic model describing your CLI options ----
class Args(BaseModel):
    scenario: Scenario = Field(
        default=Scenario.Offline,
        description="Scenario",
    )
    model_path: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Model name"
    )
    dataset_path: str = Field(
        default="",
        description="Path to dataset"
    )
    accuracy: Annotated[bool, typer.Option()] = Field(
        default=False,
        description="Run accuracy mode",
    )
    dtype: DType = Field(
        default="float32",
        description="Data type of the model, choose from float16, bfloat16 and float32",
    )
    audit_conf: str = Field(
        default="audit.conf",
        description="Audit config for LoadGen settings during compliance runs",
    )
    user_conf: str = Field(
        default="user.conf",
        description="User config for LoadGen settings such as target QPS",
    )
    total_sample_count: int = Field(
        default=13368,
        description="Number of samples to use in benchmark.",
    )
    batch_size: int = Field(
        default=1,
        description="Model batch-size to use in benchmark.",
    )
    output_log_dir: str = Field(
        default="output",
        description="Where logs are saved",
    )
    enable_log_trace: Annotated[bool, typer.Option()] = Field(
        default=False,
        description="Enable log tracing (file can become large)",
    )
    num_workers: int = Field(
        default=1,
        description="Number of workers to process queries",
    )
    tensor_parallel_size: int = Field(
        default=8,
        description="Tensor-parallel size",
    )
    vllm: Annotated[bool, typer.Option()] = Field(
        default=False,
        description="Enable vLLM mode",
    )
    api_model_name: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        description="Model name (as exposed by the LLM server)",
    )
    api_server: Optional[str] = Field(
        default=None,
        description="API endpoint to use API mode",
    )
    lg_model_name: Literal["qwen2_5-7B", "qwen2_5-7B-edge"] = Field(
        default="qwen2_5-7B",
        description="LoadGen-visible model name",
    )

scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream,
}


def main(args: Args):

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    if args.vllm:
        from SUT_VLLM import SUT, SUTServer
    else:
        raise NotImplementedError

    sut_map = {"offline": SUT, "server": SUTServer, "singlestream": SUTServer}
    log.info(f"SCENARIO is : {args.scenario.lower()}")
    sut_cls = sut_map[args.scenario.lower()]

    sut = sut_cls(
        model_path=args.model_path,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        workers=args.num_workers,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(
        lgSUT,
        sut.qsl,
        settings,
        log_settings,
        args.audit_conf)

    # # Stop sut after completion
    sut.stop()

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    pydantic_typer.run(main)
