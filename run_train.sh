# python train.py ./models/iter_mask/plainvit_huge448_cocolvis_itermask.py \
# --batch-size=32 \
# --ngpus=4

# python train.py ./models/iter_mask/trimap_huge448.py \
# --batch-size=32 \
# --ngpus=4 \
# --upsample='x4'

# python train.py ./models/loss/trimap_huge448_CE_loss.py \
# --batch-size=28 \
# --ngpus=2 \
# --upsample='x4'

python train.py ./models/loss/alphamatte_huge448_BCE_loss.py \
--batch-size=28 \
--ngpus=2 \
--upsample='x4'