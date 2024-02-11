python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt pretrain/stablesr_000117.ckpt --vqgan_ckpt pretrain/vqgan_cfw_00011.ckpt --init-img INPUT_PATH --outdir output --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain

# train SFT
python main.py --train --base configs/stableSRNew/v2-finetune_text_T_512.yaml --gpus 0, --name my_color --scale_lr False

python main.py --train --base configs/stableSRNew/v2-finetune_text_T_512_color.yaml --gpus 0, --scale_lr False --resume logs/_resume_test/checkpoints/last.ckpt --gpus 0, --scale_lr False  --dist False 
CUDA_VISIBLE_DEVICES=1 python main.py --train --base configs/stableSRNew/v2-finetune_text_T_512_color.yaml --gpus 0, --scale_lr False  --dist False 


# train CFW
# General SR
python scripts/generate_vqgan_data.py --config configs/stableSRdata/test_data.yaml --ckpt checkpoints/last.ckpt --outdir OUTDIR --skip_grid --ddpm_steps 200 --base_i 0 --seed 10000

CUDA_VISIBLE_DEVICES=0 python scripts/generate_vqgan_data.py --config configs/stableSRdata/test_data_color.yaml --ckpt logs/2024-01-11T04-32-41_color_test/checkpoints/epoch=000004-v19.ckpt --outdir OUTDIR --skip_grid --ddpm_steps 200 --base_i 0 --seed 10000

then
    # data folder 
    CFW_trainingdata
        └── inputs
            └── 00000001.png # LQ images, (512, 512, 3) (resize to 512x512)
            └── ...
        └── gts
            └── 00000001.png # GT images, (512, 512, 3) (512x512)
            └── ...
        └── latents
            └── 00000001.npy # Latent codes (N, 4, 64, 64) of HR images generated by the diffusion U-net, saved in .npy format.
            └── ...
        └── samples
            └── 00000001.png # The HR images generated from latent codes, just to make sure the generated latents are correct.
            └── ...

then
python main.py --train --base configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml --gpus 0, --name my_color --scale_lr False


# test
python scripts/sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt CKPT_PATH --vqgan_ckpt VQGANCKPT_PATH --init-img INPUT_PATH --outdir OUT_DIR --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain

CUDA_VISIBLE_DEVICES=0 python scripts/sr_val_ddpm_text_T_vqganfin_color.py --config configs/stableSRNew/v2-finetune_text_T_512_color.yaml --ckpt logs/2024-01-11T04-32-41_color_test/checkpoints/epoch=000004-v19.ckpt --vqgan_ckpt pretrain/vqgan_cfw_00011.ckpt --init-img INPUT_PATH --outdir output_color --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain


# resume
python main.py --train --base configs/stableSRNew/v2-finetune_text_T_512.yaml --gpus 0, --resume pretrain/stablesr_000117.ckpt --scale_lr False