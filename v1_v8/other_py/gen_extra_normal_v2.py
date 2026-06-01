"""Generate 1500 more normal rows to enable S15k + fix sf_auc in pipeline."""
import random, re, pandas as pd
from pathlib import Path
random.seed(55555)
BASE = Path("c:/Users/chanh/OneDrive/바탕 화면/4차 결과물")
OUT  = BASE / "pipeline_output_50k_v2"

KO_RE = re.compile(r"[가-힣]")
def has_ko(s): return bool(KO_RE.search(str(s)))
def lbin(n):
    if n<100: return "20_99"
    if n<200: return "100_199"
    if n<300: return "200_299"
    return "300_399"
def norm_t(s):
    s=str(s).lower().strip()
    s=re.sub(r"\d{3,}","NUM",s)
    s=re.sub(r"[^\w\s가-힣]"," ",s)
    return re.sub(r"\s+"," ",s).strip()[:180]

pool = pd.read_csv(OUT/"50k_v2_llm_candidate_pool_raw.csv", encoding="utf-8-sig", low_memory=False)
v1   = pd.read_csv(BASE/"final_prompt_dataset_50000_v1_train_valid_test.csv",
                   encoding="utf-8-sig", low_memory=False, usecols=["prompt"])
all_seen = set(pool["prompt"].apply(norm_t)) | set(v1["prompt"].apply(norm_t))

def dedup_gen(gen_fn, n, mult=20):
    seen=set(all_seen); res=[]
    for _ in range(n*mult):
        if len(res)>=n: break
        t=gen_fn().strip()
        if not t or not has_ko(t) or len(t)<20 or len(t)>=700: continue
        nm=norm_t(t)
        if nm not in seen: seen.add(nm); all_seen.add(nm); res.append(t)
    return res

def make_row(p,sd,sf_style,grp,idx):
    ln=len(p)
    return {"prompt":p,"label":0,"label_name":"normal","source_detail":sd,
            "source_family":"llm_generated_pool","source_group":"normal",
            "origin_type":"template_generated_v2","style_family":sf_style,
            "attack_type":"none","risk_subtype":"general_normal",
            "is_hard_negative":False,"normal_category":"general_normal",
            "replacement_role":"","length":ln,"length_bin":lbin(ln),
            "pair_id":f"llm_v2x_{grp}_{idx:05d}","split_group_id":f"llm_v2x_{grp}_{idx:05d}",
            "quality_flags":"","file_source":"llm_generated_v2","generation_group":grp}

CORPS = ["삼성전자","LG전자","카카오","네이버","SK텔레콤","KT","현대자동차","롯데","포스코","신한은행",
         "KB국민은행","우리은행","하나금융","쿠팡","배달의민족","토스","당근마켓","무신사","마켓컬리"]
ITEMS = ["스마트폰","노트북","태블릿","에어팟","청소기","냉장고","세탁기","TV"]
FOODS = ["비빔밥","삼겹살","치킨","피자","파스타","스시","라면","떡볶이"]
PLACES = ["서울","부산","제주","강원도","경주","전주","인천","대구"]
TOOLS  = ["파이썬","엑셀","SQL","노션","슬랙","지라","피그마","깃허브"]

def gen_short_v2():
    patterns = [
        f"{random.choice(CORPS)} 주가 오늘 어때요?",
        f"{random.choice(ITEMS)} {random.randint(30,200)}만원대 추천해줘.",
        f"{random.choice(FOODS)} 레시피 알려줘.",
        f"{random.choice(PLACES)} {random.randint(1,5)}박 여행 코스 짜줘.",
        f"{random.choice(TOOLS)} 기초 배우는 방법 알려줘.",
        f"오늘 {random.choice(['주식','코인','ETF'])} 시장 동향 어때요?",
        f"{random.choice(['취업','이직'])} 준비 어디서부터 시작해요?",
        f"{random.choice(['유산소','근력','스트레칭'])} 운동 루틴 추천해줘.",
        f"{random.choice(['봄','여름','가을','겨울'])} {random.choice(['코디','음식','여행'])} 추천해줘.",
        f"신입 개발자 {random.choice(['연봉','면접','포트폴리오'])} 궁금해요.",
        f"{random.choice(['커피','차','주스'])} 집에서 만드는 법 알려줘.",
        f"{random.randint(20,60)}대 {random.choice(['남성','여성'])} {random.choice(['헬스','다이어트','건강관리'])} 팁은?",
        f"요즘 {random.choice(['인기','핫한','떠오르는'])} {random.choice(['앱','서비스','기술'])} 뭐예요?",
        f"{random.choice(CORPS)} 신입 공채 일정 언제예요?",
        f"MBTI {random.choice(['INFP','ENFJ','ISTP','ESTJ'])}랑 잘 맞는 직업은?",
    ]
    return random.choice(patterns)

def gen_biz_v2():
    patterns = [
        f"{random.choice(CORPS)} 신제품 홍보 문구 작성해 주세요. 타깃: {random.randint(20,50)}대. 채널: {random.choice(['SNS','이메일','블로그'])}.",
        f"다음 주 {random.choice(['월','화','수','목','금'])}요일 팀 회의 안건 정리해 주세요. 주제: {random.choice(['Q4 계획','예산 검토','프로젝트 진행'])}.",
        f"고객 불만 답변 작성해 주세요. 내용: {random.choice(['배송 지연','상품 하자','환불 요청'])}. 상대: 장기 고객.",
        f"업무 이메일 교정해 주세요. 내용: {random.choice(['미팅 요청','계약서 첨부','결과 보고'])}. 수신: {random.choice(['협력사','고객사','파트너'])}.",
        f"제품 설명서 요약해 주세요. {random.choice(['전문 용어 제거','핵심만 3줄로','초등학생도 이해하도록'])} 부탁드립니다.",
        f"다음 보고서 초안을 검토해 주세요. 주요 지표: 매출 {random.randint(10,500)}억, 성장률 {random.randint(-10,50)}%.",
    ]
    return random.choice(patterns)

new_rows = []
idx = 20000
for gen_fn, n, sd, style, grp in [
    (gen_short_v2, 800, "llm_v2_short_gn_extra","short_general_question","short_gn_extra"),
    (gen_biz_v2,   700, "llm_v2_biz_normal_extra","business_task","biz_normal_extra"),
]:
    rows = dedup_gen(gen_fn, n)
    for p in rows:
        new_rows.append(make_row(p, sd, style, grp, idx)); idx+=1
    print(f"  {grp}: {len(rows)}")

new_df = pd.DataFrame(new_rows)
combined = pd.concat([pool, new_df], ignore_index=True)
combined.to_csv(OUT/"50k_v2_llm_candidate_pool_raw.csv", index=False, encoding="utf-8-sig")

atk = int((combined["label"]==1).sum())
nrm = int((combined["label"]==0).sum())
print(f"Final v2 pool: {len(combined)} (attack={atk}, normal={nrm})")
