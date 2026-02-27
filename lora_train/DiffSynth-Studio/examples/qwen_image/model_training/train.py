import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *
import torch.nn as nn
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None, lora_dropout: float = 0.0,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
        use_3d_loss=False,
        three_d_loss_weight=1.0,
        three_d_qwen_layers="48",
        three_d_timestep_max=None,
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device
        )
        tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)

        self.lora_dropout = lora_dropout

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config
        )
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
            lora_dropout=self.lora_dropout,
        )

        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        from vggt_align_loss import VGGTGeometryForcingLoss

        self.use_3d_loss = False  # default
        self.three_d_loss_weight = 1.0
        self.three_d_qwen_layers = []
        self._qwen_hook_handles = []
        self._qwen_hidden_cache = {}
        self.use_3d_loss = use_3d_loss
        self.three_d_loss_weight = three_d_loss_weight
        self.three_d_qwen_layers = [int(x) for x in three_d_qwen_layers.split(",") if x.strip()]

        if self.use_3d_loss:
            self.vggt_loss = VGGTGeometryForcingLoss(
                device=self.pipe.device,
                dtype=self.pipe.torch_dtype,
                qwen_layer_indices=self.three_d_qwen_layers,
                timestep_max_for_3d=three_d_timestep_max,
            )
            self._install_qwen_hooks()
        else:
            self.vggt_loss = None

        # diese Werte kommen aus args, du kannst sie im __init__ als Parameter reinreichen (empfohlen)

        self.zero_cond_t = zero_cond_t

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}

        inputs_shared = {
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,
        }

        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        if isinstance(data["image"], list):
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"][0].size[1],
                "width": data["image"][0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
            })

        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if self.use_3d_loss:
            self._qwen_hidden_cache.clear()
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)

        if self.use_3d_loss and len(self._qwen_hidden_cache) > 0:
            k = next(iter(self._qwen_hidden_cache))
            if hasattr(torch,
                       "distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
                pass
            else:
                print("QWEN HOOK SHAPE:", k, self._qwen_hidden_cache[k].shape)

        if self.use_3d_loss and self.vggt_loss is not None:
            gt = data.get("image", None)

            if gt is not None:
                gt_t = self.transfer_data_to_device(gt, self.pipe.device, self.pipe.torch_dtype)

                if gt_t.dim() == 4:
                    gt_t = gt_t.unsqueeze(1)
                elif gt_t.dim() == 5:
                    pass
                else:
                    pass

                t = None

                qwen_feats = dict(self._qwen_hidden_cache)
                if len(qwen_feats) > 0:
                    loss_3d = self.vggt_loss(qwen_feats, gt_images_btc=gt_t, timesteps=t)
                    loss = loss + (self.three_d_loss_weight * loss_3d)
        return loss

    def _install_qwen_hooks(self):
        transformer = None
        candidates = []
        for name, m in self.pipe.named_modules():
            if name.endswith("transformer") or "transformer" in name or "denoiser" in name:
                candidates.append((name, m))

        if len(candidates) > 0:
            candidates.sort(key=lambda x: sum(p.numel() for p in x[1].parameters()), reverse=True)
            transformer = candidates[0][1]

        if transformer is None:
            raise RuntimeError("Could not locate transformer/denoiser module to hook for 3d loss.")

        blocks = None
        for attr in ["blocks", "layers", "transformer_blocks", "h", "model", "module"]:
            if hasattr(transformer, attr):
                obj = getattr(transformer, attr)
                if isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
                    blocks = obj
                    break

        if blocks is None:
            for _, m in transformer.named_modules():
                if isinstance(m, nn.ModuleList) and len(m) >= max(self.three_d_qwen_layers) + 1:
                    blocks = m
                    break

        if blocks is None:
            raise RuntimeError("Could not locate transformer blocks ModuleList for hooking.")

        self._qwen_hidden_cache = {}
        self._qwen_hook_handles = []

        def make_hook(layer_idx: int):
            def hook_fn(module, inp, out):
                # out kann tuple sein – wir nehmen den ersten Tensor
                if isinstance(out, (tuple, list)):
                    out0 = out[0]
                else:
                    out0 = out
                self._qwen_hidden_cache[layer_idx] = out0
            return hook_fn

        for li in self.three_d_qwen_layers:
            if li < 0 or li >= len(blocks):
                raise ValueError(f"Requested qwen layer {li} out of range (0..{len(blocks)-1}).")
            h = blocks[li].register_forward_hook(make_hook(li))
            self._qwen_hook_handles.append(h)

def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)

    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")

    parser.add_argument("--val_dataset_base_path", type=str, default=None)
    parser.add_argument("--val_dataset_metadata_path", type=str, default=None)

    parser.add_argument("--eval_loss_every_steps", type=int, default=0)
    parser.add_argument("--eval_infer_every_steps", type=int, default=0)
    parser.add_argument("--eval_num_samples", type=int, default=5)
    parser.add_argument("--eval_sample_ids", type=str, default="", help="comma-separated sample_id values to track")
    parser.add_argument("--eval_infer_steps", type=int, default=20)
    parser.add_argument("--eval_cfg_scale", type=float, default=1.0)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--eval_max_val_batches", type=int, default=0, help="0 = full val")

    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout (PEFT lora_dropout).")

    parser.add_argument("--use_3d_loss", default=False, action="store_true")
    parser.add_argument("--three_d_loss_weight", type=float, default=1.0)
    parser.add_argument("--three_d_qwen_layers", type=str, default="48",
                        help="comma-separated transformer block indices")
    parser.add_argument("--three_d_timestep_max", type=int, default=None,
                        help="optional: only apply 3d loss when t<=this")
    # -----------------------------------------

    return parser


def build_dataset(base_path: str, metadata_path: str, args, repeat: int):
    """
    Uses the newer DiffSynthStudio operators so that:
    - "image" can be either str or list (Qwen multi-image)
    - layered inputs are supported
    """
    return UnifiedDataset(
        base_path=base_path,
        metadata_path=metadata_path,
        repeat=repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
        special_operator_map={
            # Qwen-Image-Layered
            "layer_input_image": ToAbsolutePath(base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
            "image": RouteByType(operator_map=[
                (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)),
                (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16))),
            ])
        }
    )


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if os.environ.get("WANDB_PROJECT") else None,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )


    if os.environ.get("WANDB_PROJECT"):
        accelerator.init_trackers(
            project_name=os.environ["WANDB_PROJECT"],
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": os.environ.get("WANDB_RUN_NAME", None),
                    "entity": os.environ.get("WANDB_ENTITY", None),
                }
            },
        )
        print("WANDB INIT CALLED:", os.environ.get("WANDB_PROJECT"))
        print("TRACKERS:", accelerator.trackers)


    dataset = build_dataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        args=args,
        repeat=args.dataset_repeat,
    )

    val_dataset = None
    if args.val_dataset_base_path and args.val_dataset_metadata_path:
        val_dataset = build_dataset(
            base_path=args.val_dataset_base_path,
            metadata_path=args.val_dataset_metadata_path,
            args=args,
            repeat=1,
        )

    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        zero_cond_t=args.zero_cond_t,
        lora_dropout=args.lora_dropout,
        use_3d_loss=args.use_3d_loss,
        three_d_loss_weight = args.three_d_loss_weight,
        three_d_qwen_layers = args.three_d_qwen_layers,
        three_d_timestep_max = args.three_d_timestep_max,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }

    launcher_map[args.task](accelerator, dataset, model, model_logger, val_dataset=val_dataset, args=args)

    if os.environ.get("WANDB_PROJECT"):
        accelerator.end_training()