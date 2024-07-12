# Daily Paper in 3 Sentences
**Keywords**: Dataset, Education, LLM, sLM, Korean, Vocab Expansion, Paraphrase Generation, Knowledge Distillation, Fine-tuning, Controlled Generation

[2024.07.04]
### (EEVE) Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models
Archive 2024, <https://arxiv.org/pdf/2402.14714>    
_Korean | LLM | Vocab Expansion_
+ English-centric LLM의 tokenizer를 사용했을 때 한국어는 더 많은 토큰으로 분리됨 -> longer response times, shorter context lengths, higher API costs
+ 새로운 vocab이 반영된 embedding layer와 LM head를 업데이트하기 위해 6단계의 순차적 훈련 -> 모델이 기존 token과 새로운 token들을 align할 수 있도록
+ pretraining 단계에서 Transformer layer들을 업데이트하기 위해 QLoRA 활용, fine-tuning 단계에서는  DPO 적용 및 영어 intruction tuning 데이터 번역해서 활용

[2024.07.05]
### (DPO) Direct Preference Optimization: Your Language Model is Secretly a Reward Model
NeurIPS 2023, <https://arxiv.org/pdf/2305.18290>   
_Fine-tuning_   
+ RLHF는 human preference data로 reward model을 훈련 시킨 뒤, 이를 기반으로 RL을 통해 LM을 업데이트함
+ DPO는 reward model을 별도로 두지 않고, reward function을 optimial policy에 대한 수식으로 표현 후 human preference data를 이용해 바로 LM을 업데이트함
+ human preference data가 없는 controlled sentiment generation task를 위해 pretrained sentiment classifier 모델을 이용해 pair 구축함: $x$로부터 $y_1$, $y_2$를 생성한 뒤 p( $positive$ | $x$, $y_w$ ) > p( $postive$ | $x$, $y_l$ )로 preference labeling

[2024.07.12]   
### Fine-grained Gender Control in Machine Translation with Large Language Models
NAACL 2024, <https://aclanthology.org/2024.naacl-long.303.pdf>   
_LLM, Controlled Generation_   
+ 번역 src text에 포함된 ambiguous entity들 (ex, cook, lawyer)에 성별을 지정하여 번역 문장에서 gender expression들이 명확하게 사용될 수 있도록 함
+ LLama2-70B, GPT-3.5-turbo에 간단한 prompt engineering으로 해결 (gender annotation 정보를 추가로 입력)
+ LLM이 스스로 context로부터 entity들의 gender를 유추한 뒤 translation에 활용하는 CoT와 같은 방법론도 효과가 있었지만 pseudo-label gender를 명시해 주는 것보다는 낮은 성능을 보임

---------------------------------------
# + $\alpha$
[2024.07.03]   
### RACE: Large-scale Reading Comprehension Dataset From Examinations   
EMNLP 2017, <https://arxiv.org/pdf/1704.04683v5>   
_Dataset | Education_
+ 12-18세 학생을 대상으로 하는 중고등학교 영어 시험 문제들로부터 데이터 수집
+ question과 answer candidate들이 passage의 span이 아닌 경우가 많음
+ Reasoning difficutly 5단계에 따라 질문들을 분류함

[2024.07.04]
### Parameter Efficient Diverse Paraphrase Generation Using Sequence-Level Knowledge Distillation
ICACS 2024, <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10473289>   
_Paraphrase Generation | LLM | Knowledge Distillation_
+ ChatGPT를 활용한 data-centric sequence-level knowledge distillation (T5-small, Flan T5-small, BART-base 등 small LM에 LoRA까지 사용하여 computationally efficient하게 훈련)
+ ChatGPT의 temperature를 0으로 설정한 뒤 주어진 sentence의 paraphrase들을 list 형식으로 반환하게함 -> 200만 쌍의 paraphrase pair를 구축
+ Inference 단계에서 decoding hyperparameter들을 유연하게 조절 (top-p, top-k, beam size 등)

### Quality Controlled Paraphrase Generation   
ACL 2022, <https://aclanthology.org/2022.acl-long.45.pdf>   
_Paraphrase Generation_
+ Paraphrase의 Qualtity를 three dimenstion으로 구분 및 생성 조절: semantic similarity (Bleurt score), syntactic diversity (tree edit distance), lexical diversity (character-level minimal edit distance)
+ 주어진 sentence로 모든 paraphrase quality에 도달할 수 있다는 보장이 없음 (고유명사, 숫자 등) -> sentence-aware quality control
+ syntactic&lexical offset을 너무 크게 주면 semantic score가 발산하므로 dev set을 통해 "semantic score를 특정 값 이상으로 유지하면서 linguisitc diversity를 최대화하는 offset"을 선택하여 test set에 사용

[2024.07.05]
### Dictionary-Guided Editing Networks for Paraphrase Generation
AAAI 2019, <https://dl.acm.org/doi/pdf/10.1609/aaai.v33i01.33016546>   
_Paraphrase Generation_
+ Paraphrase Database (PPDB)로부터 input sentence와 유사한 word/phrase-level paraphrase pair candidate들을 검색 (TF-IDF, PPDB score 등을 통해 유사도 측정)
+ attention mechanism을 적용한 seq2seq 모델 활용 -> decoding step마다 candidate pair들과 soft attention을 통해 deletion 및 insertion을 결정
+ Paraphrase Database (PPDB)은 lexical, phrasal, syntactic type의 paraphrase들을 저장 (여러 언어를 지원하지만 한국어는 없음)

[2024.07.07]
### MCPG: A Flexible Multi-Level Controllable Framework for Unsupervised Paraphrase Generation
EMNLP Findings 2022, <https://aclanthology.org/2022.findings-emnlp.439.pdf>   
_Paraphrase Generation_
+ input sentence의 semantic embedding의 variation들을 Encoder의 dropout probability를 통해 조절 (dropout 전후의 embedding의 cosine similarity가 0.75 이상인 경우만 의미가 유지됨)
+ T5를 기반으로 input sentence의 named entity들 사이의 special token들을 infilling하는 task로 접근 -> T5 encoder의 output representation 앞에 BERT의 drop-out을 거친 semantic embedding을 concat하여 decoding
+ target domain의 paraphrase pair들을 활용할 수 있는 경우, input sentence와 cosine similarity가 가장 높은 pair를 찾아 pair 간의 style 차이를 input sentence embedding에 더하여 활용 (Style transfer)

### Impossible Distillation for Paraphrasing and Summarization: How to Make High-quality Lemonade out of Small, Low-quality Models
NAACL 2024, <https://aclanthology.org/2024.naacl-long.250.pdf>   
_Paraphrase Generation | sLM | Knowledge Distilation_   
+ informative context가 주어졌을 때 LM은 서로 paraphrase되는 여러 문장들을 생성할 수 있음 (necleus sampling 활용)
+ semantic equivalence filter (NLI), dissimilarity filter (ROUGE-L, TED), diversity filter (NLI)를 통해 생성된 paraphrase pool로부터 구성된 pair들을 필터링
+ teacher (GPT2-XL, 1.5B) 모델로 생성한 데이텨로 student (T5-large) 모델을 훈련 & self-distillation (훈련된 student 모델의 inference 결과를 필터링 후 다시 훈련 데이터로 활용)

[2024.07.08]
### (BERT-iBLEU) Unsupervised Paraphrasing with Pretrained Language Models
EMNLP 2021, <https://aclanthology.org/2021.emnlp-main.417.pdf>   
_Paraphrase Generation_
+ BERT-iBLEU score: semantic similarity가 높으면서 surface-form similarity가 낮을 때 높은 점수를 부여하는 metric (paraphrase generation 논문들에서 많이 활용되고 있음)
+ supervised learning이 되지 않은 pretrained LM을 활용해 paraphrase를 생성하기 위해 decoding 단계에 constraint를 부여 -> Dynamic Blocking
+ Dynamic Blocking은 source text의 ($s_i$, $s_{i+1}$)들을 확률적으로 sampling한 뒤 decoding의 $j$-step에서 $s_i$가 생성되면 $j+1$-step에서 $s_{i+1}$의 probability를 0으로 만듦
