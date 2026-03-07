cuda=${1:-0}
init_model=init
init_iter=40000
iter=20000
model=radiogs
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR_Ind/armadillo \
    -m outputs/TensoIR_Ind/armadillo/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_radiosity 0.1 \
    --lambda_normal_smooth 0.02 \

# ficus
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR_Ind/ficus \
    -m outputs/TensoIR_Ind/ficus/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_normal_smooth 0.02 \
    --lambda_radiosity 0.1 

# hotdog
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR_Ind/hotdog \
    -m outputs/TensoIR_Ind/hotdog/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_normal_smooth 0.02 \
    --lambda_radiosity 0.1 

# lego
CUDA_VISIBLE_DEVICES=$cuda python train_init.py -s data/TensoIR_Ind/lego \
    -m outputs/TensoIR_Ind/lego/$init_model --eval -w \
    --lambda_mask_entropy 0.02 \
    --volume_render_until_iter $init_iter \
    --iterations $init_iter \
    --lambda_dist 1000 \
    --lambda_light 0.01 \
    --train_sh_vol \
    --radiosity_from_iter 2000 \
    --lambda_normal_smooth 0.02 \
    --lambda_radiosity 0.1 

# armadillo
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR_Ind/armadillo --eval \
    -m outputs/TensoIR_Ind/armadillo/$model --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR_Ind/armadillo/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.02 --lambda_light 0.01 \
    --lr_scale 0.01 --lambda_nvs 1.0 --back_culling --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR_Ind/armadillo/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_indirect.py -m outputs/TensoIR_Ind/armadillo/$model
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Ind/armadillo/$model   --back_culling \
    -s data/TensoIR_Ind/armadillo
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/TensoIR_Ind/armadillo/$model \
    --albedo_rescale 2 -s data/TensoIR_Ind/armadillo   --back_culling


# hotdog
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR_Ind/hotdog --eval \
    -m outputs/TensoIR_Ind/hotdog/$model --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR_Ind/hotdog/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.02 --lambda_light 0.01 \
    --lr_scale 0.01 --lambda_nvs 1.0  --back_culling --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \
    --light_t_min 0.1

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR_Ind/hotdog/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling --light_t_min 0.1 
CUDA_VISIBLE_DEVICES=$cuda python eval_indirect.py -m outputs/TensoIR_Ind/hotdog/$model
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Ind/hotdog/$model   --back_culling \
    -s data/TensoIR_Ind/hotdog
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/TensoIR_Ind/hotdog/$model --albedo_rescale 2   --back_culling \
    -s data/TensoIR_Ind/hotdog

# ficus
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR_Ind/ficus --eval \
    -m outputs/TensoIR_Ind/ficus/$model  --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR_Ind/ficus/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128  --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.6 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.002 --lambda_light 0.01 \
    --lr_scale 0.01 --lambda_nvs 1.0  --back_culling  --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR_Ind/ficus/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_indirect.py -m outputs/TensoIR_Ind/ficus/$model
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Ind/ficus/$model   --back_culling \
    -s data/TensoIR_Ind/ficus
CUDA_VISIBLE_DEVICES=$cuda python eval_material_tensoir.py -m outputs/TensoIR_Ind/ficus/$model \
    --albedo_rescale 2 -s data/TensoIR_Ind/ficus   --back_culling

# lego
CUDA_VISIBLE_DEVICES=$cuda python train.py -s data/TensoIR_Ind/lego --eval \
    -m outputs/TensoIR_Ind/lego/$model --iterations $iter \
    --start_checkpoint_refgs outputs/TensoIR_Ind/lego/$init_model/chkpnt$init_iter.pth \
    --envmap_resolution 128 --diffuse_sample_num 64 --envmap_cubemap_lr 0.005 --init_roughness_value 0.8 \
    --lambda_base_color_smooth 0.2 --lambda_roughness_smooth 0.1 --lambda_light_smooth 0.02 --lambda_light 0.1 \
    --lr_scale 0.01 --lambda_nvs 1.0  --back_culling --lambda_mask_entropy 0.02 \
    --use_radiosity --lambda_radiosity 0.2 --radiosity_gaussian_num 2048 --radiosity_sample_num 64 \
    --use_rad_rndview --detach_rad_global \

CUDA_VISIBLE_DEVICES=$cuda python render.py -m outputs/TensoIR_Ind/lego/$model --eval --skip_train \
    --diffuse_sample_num 64   --back_culling
CUDA_VISIBLE_DEVICES=$cuda python eval_indirect.py -m outputs/TensoIR_Ind/lego/$model
CUDA_VISIBLE_DEVICES=$cuda python compute_albedo_scale_tensoir.py -m outputs/TensoIR_Ind/lego/$model   --back_culling \
    -s data/TensoIR_Ind/lego