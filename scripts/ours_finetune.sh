cuda=${1:-0}
model=radiogs
iter=20000
finetune_iter=22000
tag=finetune_nvs01

# output of finetuned model
armadillo_model=radiogs
ficus_model=radiogs
hotdog_model=radiogs
lego_model=radiogs

envmaps="bridge city fireplace forest night"

for em in $envmaps; do


    finetune_model_em=$armadillo_model/$tag/envmap_$em
    # armadillo
    CUDA_VISIBLE_DEVICES=$cuda python finetune.py -s data/TensoIR/armadillo --eval \
        -m outputs/TensoIR/armadillo/$finetune_model_em --iterations $finetune_iter \
        --start_checkpoint outputs/TensoIR/armadillo/$model/chkpnt$iter.pth --envmap_resolution 128 --diffuse_sample_num 192 \
        --lr_scale 0.0  \
        --use_radiosity --lambda_radiosity 1.0 --radiosity_gaussian_num 8192 --radiosity_sample_num 128 --light_sample_num 64 --use_rad_imp \
        --use_rad_rndview --detach_rad_mat --detach_rad_normal --lambda_normal_smooth 0.0 --detach_rad_global \
        --envmap_path data/Environment_Maps/high_res_envmaps_2k/$em.hdr --radiosity_random_sample 1 \
         --back_culling --features_lr 1e-4 --lambda_nvs 0.1

    finetune_model_em=$ficus_model/$tag/envmap_$em
    # ficus
    CUDA_VISIBLE_DEVICES=$cuda python finetune.py -s data/TensoIR/ficus --eval \
        -m outputs/TensoIR/ficus/$finetune_model_em --iterations $finetune_iter \
        --start_checkpoint outputs/TensoIR/ficus/$model/chkpnt$iter.pth --envmap_resolution 128 --diffuse_sample_num 192 \
        --lr_scale 0.0  --back_culling \
        --use_radiosity --lambda_radiosity 1.0 --radiosity_gaussian_num 8192 --radiosity_sample_num 128 --light_sample_num 64 --use_rad_imp \
        --use_rad_rndview --detach_rad_mat --detach_rad_normal --lambda_normal_smooth 0.0 --detach_rad_global \
        --envmap_path data/Environment_Maps/high_res_envmaps_2k/$em.hdr --radiosity_random_sample 1 \
         --back_culling --features_lr 1e-4 --lambda_nvs 0.1

    finetune_model_em=$hotdog_model/$tag/envmap_$em
    # hotdog
    CUDA_VISIBLE_DEVICES=$cuda python finetune.py -s data/TensoIR/hotdog --eval \
        -m outputs/TensoIR/hotdog/$finetune_model_em --iterations $finetune_iter \
        --start_checkpoint outputs/TensoIR/hotdog/$model/chkpnt$iter.pth --envmap_resolution 128 --diffuse_sample_num 192 \
        --lr_scale 0.0  --light_t_min 0.1 \
        --use_radiosity --lambda_radiosity 1.0 --radiosity_gaussian_num 8192 --radiosity_sample_num 128 --light_sample_num 64 --use_rad_imp \
        --use_rad_rndview --detach_rad_mat --detach_rad_normal --lambda_normal_smooth 0.0 --detach_rad_global \
        --envmap_path data/Environment_Maps/high_res_envmaps_2k/$em.hdr --radiosity_random_sample 1 \
         --features_lr 1e-4 --lambda_nvs 0.1

    finetune_model_em=$lego_model/$tag/envmap_$em
    # lego
    CUDA_VISIBLE_DEVICES=$cuda python finetune.py -s data/TensoIR/lego --eval \
        -m outputs/TensoIR/lego/$finetune_model_em --iterations $finetune_iter \
        --start_checkpoint outputs/TensoIR/lego/$model/chkpnt$iter.pth --envmap_resolution 128 --diffuse_sample_num 192 \
        --lr_scale 0.0  --back_culling \
        --use_radiosity --lambda_radiosity 1.0 --radiosity_gaussian_num 8192 --radiosity_sample_num 128 --light_sample_num 64 --use_rad_imp \
        --use_rad_rndview --detach_rad_mat --detach_rad_normal --lambda_normal_smooth 0.0 --detach_rad_global \
        --envmap_path data/Environment_Maps/high_res_envmaps_2k/$em.hdr --radiosity_random_sample 1 \
         --back_culling --features_lr 1e-4 --lambda_nvs 0.1

done


CUDA_VISIBLE_DEVICES=$cuda python finetune_relighting_tensoir.py -m outputs/TensoIR/armadillo/$armadillo_model \
        --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 4   --back_culling \
        --finetune --iteration $finetune_iter --extra $tag

CUDA_VISIBLE_DEVICES=$cuda python finetune_relighting_tensoir.py -m outputs/TensoIR/ficus/$ficus_model \
        --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2   --back_culling \
        --finetune --iteration $finetune_iter --extra $tag

CUDA_VISIBLE_DEVICES=$cuda python finetune_relighting_tensoir.py -m outputs/TensoIR/hotdog/$hotdog_model \
        --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2   --back_culling \
        --finetune --iteration $finetune_iter --extra $tag

CUDA_VISIBLE_DEVICES=$cuda python finetune_relighting_tensoir.py -m outputs/TensoIR/lego/$lego_model \
        --diffuse_sample_num 256 --light_sample_num 128 --albedo_rescale 2   --back_culling \
        --finetune --iteration $finetune_iter --extra $tag
