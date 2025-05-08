## 수정사항

1. is_trimap_plaintvit_model.py : pos_embed 문제를 해결하지 않고 그대로 적용
2. is_trimap_plaintvit_model_noposembed.py : pos_embed 문제 해결된 코드 (테스트는 아직 안해봄). 이걸 사용하려면 train 코드에서 호출할때 NoPosEmbedTrimapPlainVitModel 클래스를 호출하도록 수정해야함. 그외에는 수정 필요 없음
3. is_alphamatte_plaintvit_model.py : alphamatte를 추정하도록 구조 수정중
