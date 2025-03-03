# GroundingDINO_SwinB_cfg.py 파일의 설정 값들에 대한 설명

: 이 설정들은 모델의 구성 및 학습과 관련된 다양한 하이퍼파라미터와 옵션을 정의

(이하는 gpt한테 설명을 부탁한 내용)

1. batch_size: 배치 크기. 여기서는 1로 설정되어 있습니다.
2. modelname: 사용되는 모델의 이름입니다. 여기서는 "groundingdino"로 설정되어 있습니다.
3. backbone: 백본 모델의 이름. 여기서는 "swin_B_384_22k"로 설정되어 있습니다.
4. position_embedding: 위치 임베딩의 종류입니다. 여기서는 "sine"으로 설정되어 있습니다.
5. pe_temperatureH & pe_temperatureW: 위치 임베딩의 온도입니다.
6. return_interm_indices: 반환되는 중간 인덱스입니다.
7. backbone_freeze_keywords: 백본 모델의 특정 부분을 고정하기 위한 키워드입니다. 현재는 아무 것도 설정되어 있지 않습니다.
8. enc_layers & dec_layers: 인코더와 디코더의 레이어 수입니다.
9. pre_norm: 사전 정규화의 사용 여부입니다. 현재는 False로 설정되어 있습니다.
10. dim_feedforward: Feedforward 네트워크의 차원입니다.
11. hidden_dim: 히든 레이어의 차원입니다.
12. dropout: 드롭아웃 비율입니다.
13. nheads: 멀티 헤드 어텐션의 헤드 수입니다.
14. num_queries: 쿼리의 수입니다.
15. query_dim: 쿼리의 차원입니다.
16. num_patterns: 패턴의 수입니다. 현재는 0으로 설정되어 있습니다.
17. num_feature_levels: 특징 레벨의 수입니다.
18. enc_n_points & dec_n_points: 인코더와 디코더의 포인트 수입니다.
19. two_stage_type: 두 단계 모델의 유형입니다. 여기서는 "standard"로 설정되어 있습니다.
20. two_stage_bbox_embed_share: 두 단계 모델에서 Bounding Box 임베딩을 공유하는지 여부입니다.
21. two_stage_class_embed_share: 두 단계 모델에서 클래스 임베딩을 공유하는지 여부입니다.
22. transformer_activation: 트랜스포머에서 사용되는 활성화 함수입니다. 여기서는 "relu"로 설정되어 있습니다.
23. dec_pred_bbox_embed_share: 디코더에서 예측된 Bounding Box 임베딩을 공유하는지 여부입니다.
24. dn_box_noise_scale: Noise scale for the bounding box.
25. dn_label_noise_ratio: 라벨에 대한 노이즈 비율입니다.
26. dn_label_coef & dn_bbox_coef: 라벨과 Bounding Box에 대한 계수입니다.
27. embed_init_tgt: 타겟 임베딩을 초기화하는지 여부입니다.
28. dn_labelbook_size: 라벨북의 크기입니다.
29. max_text_len: 텍스트의 최대 길이입니다.
30. text_encoder_type: 텍스트 인코더의 유형입니다. 여기서는 BERT의 "bert-base-uncased" 버전을 사용합니다.
31. use_text_enhancer: 텍스트 enhancer를 사용하는지 여부입니다.
32. use_fusion_layer: 퓨전 레이어를 사용하는지 여부입니다.
33. use_checkpoint: 체크포인트를 사용하는지 여부입니다.
34. use_transformer_ckpt: 트랜스포머의 체크포인트를 사용하는지 여부입니다.
35. use_text_cross_attention: 텍스트 크로스 어텐션을 사용하는지 여부입니다.
36. text_dropout: 텍스트에 대한 드롭아웃 비율입니다.
37. fusion_dropout: 퓨전에 대한 드롭아웃 비율입니다.
38. fusion_droppath: 퓨전의 droppath 비율입니다.
39. sub_sentence_present: 하위 문장의 존재 여부입니다.
