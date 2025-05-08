## 수정사항

1. trimap_aim500_am2k_huge448_nfl_dt_loss.py : dataset이 aim500, am2k만 쓰고, loss는 focal loss와 dt loss를 사용함. 그러나 성능이 별로라 필요없음
2. trimap_huge448_nfl_dt_loss.py : 모든 데이터셋을 다 썼지만, loss는 focal loss와 dt loss를 사용함. 그러나 성능이 별로라 필요없음
3. trimap_huge448_CE_loss.py : 모든 데이터셋 다 쓰고, CE loss로만 training함. 이게 성능이 젤 좋음
4. alphamatte_huge448_CE_loss.py : alphamatte를 추론하기 위해 구조 수정함. 단순 입출력과 loss만 수정함. dataset 싱크 맞춰야하고, dataloader도 새로 만들어야하고, loss도 동작하는지 확인, validation metrics도 새롭게 만들어서 넣어야함
