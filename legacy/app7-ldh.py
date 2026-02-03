import streamlit as st
import streamlit.components.v1 as components

from src.chain.rag_chain import build_rag_chain_with_sources, prepare_context, stream_answer
from src.config import CLASSIFIER_MODEL, LLM_MODEL
from langchain_core.prompts import ChatPromptTemplate
from src.chain.prompts import ANSWER_PROMPT

# 페이지 설정
st.set_page_config(
    page_title="의약품 정보 Q&A",
    page_icon="💊",
    layout="wide",
)


# --- [수정 부분 1: CSS 스타일] ---
st.markdown("""
    <style>
    .chat-bubble {
        background-color: white;
        padding: 15px 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        display: inline-block;
        color: black;
        font-family: sans-serif;
        white-space: pre-wrap; 
        box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        word-break: break-all;
        line-height: 1.6; 
    }
    .user-message-group {
        display: flex;
        align-items: flex-start;
        justify-content: flex-end; 
        gap: 10px;
        width: 100%;
        margin-bottom: 20px;
    }
    .user-icon {
        width: 35px;
        height: 35px;
        background-color: #FF4B4B;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        order: 2; 
    }
    .user-bubble-container { order: 1; }
    </style>
""", unsafe_allow_html=True)


# --- [다국어 지원: 언어별 텍스트] ---
LANG_TEXTS = {
    "KR": {
        "title": "💊 의약품 정보 Q&A",
        "caption": "식품의약품안전처 e약은요 + 허가정보 데이터 기반 시스템",
        "placeholder": "의약품에 대해 궁금한 점을 질문해주세요...",
        "searching": "정보를 검색하고 있습니다...",
        "analysis_result": "💉분석 결과",
        "contraindication_title": "⚠️ 병용금지 주의 약물 목록",
        "contraindication_text": "과 함께 복용하면 안 되는 성분:",
        "sources_title": "📋 참고 자료 보기",
        "related_drugs": "📋 관련 의약품 정보",
        "company": "업체",
        "code": "코드",
        "search": "🔍 검색",
        "sidebar_title": "의약품 정보 Q&A 시스템",
        "sidebar_guide": "이 시스템은 식품의약품안전처 공공데이터의 의약품 정보를 제공합니다.",
        "example_title": "📝 질문 예시 (클릭하여 복사):",
        "example1": "💊 타이레놀의 효능은 무엇인가요?",
        "example2": "🧪 아세트아미노펜이 포함된 약은?",
        "example3": "🩹 두통에 효과있는 약은?",
        "warning": "⚠️ 이 시스템은 일반적인 의약품 정보를 제공하며, 의학적 진단이나 처방을 대체하지 않습니다. 반드시 의사 또는 약사와 상담하세요.",
        "reset": "대화 초기화",
    },
    "EN": {
        "title": "💊 Medication Information Q&A",
        "caption": "Based on MFDS e약은요 + Drug Approval Information",
        "placeholder": "Ask questions about medications...",
        "searching": "Searching for information...",
        "analysis_result": "💉Analysis Result",
        "contraindication_title": "⚠️ Contraindicated Medications",
        "contraindication_text": " should not be taken with:",
        "sources_title": "📋 Reference Materials",
        "related_drugs": "📋 Related Medication Information",
        "company": "Company",
        "code": "Code",
        "search": "🔍 Search",
        "sidebar_title": "Medication Information Q&A System",
        "sidebar_guide": "This system provides medication information based on MFDS public data.",
        "example_title": "📝 Example Questions (Click to copy):",
        "example1": "💊 What is the efficacy of Tylenol?",
        "example2": "🧪 What medications contain Acetaminophen?",
        "example3": "🩹 What medications are effective for headaches?",
        "warning": "⚠️ This system provides general medication information and does not replace medical diagnosis or prescription. Please consult with a doctor or pharmacist.",
        "reset": "Reset Conversation",
    },
    "JP": {
        "title": "💊 医薬品情報 Q&A",
        "caption": "韓国食品医薬品安全処 e약은요 + 医薬品承認情報に基づくシステム",
        "placeholder": "医薬品について質問してください...",
        "searching": "情報を検索しています...",
        "analysis_result": "💉分析結果",
        "contraindication_title": "⚠️ 併用禁忌薬物リスト",
        "contraindication_text": "と併用してはいけない成分:",
        "sources_title": "📋 参考資料",
        "related_drugs": "📋 関連医薬品情報",
        "company": "会社",
        "code": "コード",
        "search": "🔍 検索",
        "sidebar_title": "医薬品情報 Q&A システム",
        "sidebar_guide": "このシステムは韓国食品医薬品安全処の公開データに基づく医薬品情報を提供します。",
        "example_title": "📝 質問例（クリックしてコピー）:",
        "example1": "💊 タイレノールの効能は何ですか？",
        "example2": "🧪 アセトアミノフェンが含まれている薬は？",
        "example3": "🩹 頭痛に効果のある薬は？",
        "warning": "⚠️ このシステムは一般的な医薬品情報を提供し、医学的診断や処方を代替するものではありません。必ず医師または薬剤師にご相談ください。",
        "reset": "会話をリセット",
    }
}


# --- [다국어 지원: 언어별 프롬프트 생성] ---
def get_answer_prompt_with_language(language: str):
    """언어에 맞는 답변 생성 프롬프트를 반환합니다."""
    lang_instructions = {
        "KR": """당신은 한국 의약품 정보 전문 AI 챗봇입니다.
식품의약품안전처의 e약은요, 의약품 허가정보 데이터의 내용만으로
사용자의 질문에 정확하고 친절하게 답변합니다.

[반드시 지켜야 할 규칙]
1. 검색해서 얻은 내용을 바탕으로 성분명과 효능을 매칭시킵니다.
2. 사용자의 증상에 가장 적합한 성분명을 찾습니다.
3. 해당 성분명과 해당 성분의 대표적 효능을 답변합니다.
4. 답변은 친절하지만 방어적으로 표현합니다.
5. 사용자를 "사용자"라고 지칭합니다.
6. 공감과 같은 불필요한 표현은 하지 않습니다.
7. [다중 증상 처리]: 키워드가 여러 개(예: 요통, 두통)이고 이를 동시에 만족하는 약이 없다면, 각각의 증상에 맞는 약 정보를 구분하여 답변하십시오. (예: "요통에는 A가, 두통에는 B가 적합합니다.")
8. 검색 결과(context)가 사용자의 실제 증상과 논리적으로 일치하는지 확인하십시오.
   - 예: 사용자는 "딸깍"이라고만 했는데 검색 결과가 "관절염"이라면, 이는 잘못된 매칭이므로 약을 추천하지 말고 다시 질문해달라고 요청하십시오.

[출력 형식 - 반드시 아래 형식을 따르세요]

성분명: (해당 성분명)
효능: (해당 효능)

병용금지 주의: (DUR 정보에 병용금지 약물이 있으면 반드시 작성)
- (성분명)과 함께 복용하면 안 되는 약물: (병용금지 성분명) - (사유)

제안: (복용 관련 제안사항)""",
        "EN": """You are a professional AI chatbot specializing in Korean medication information.
Based solely on the content from MFDS e약은요 and drug approval information data,
answer users' questions accurately and kindly.

[Rules to Follow]
1. Match ingredient names and efficacy based on the search results.
2. Find the most suitable ingredient name for the user's symptoms.
3. Provide the ingredient name and its representative efficacy.
4. Be friendly but defensive in your responses.
5. Refer to the user as "user".
6. Avoid unnecessary expressions like empathy.
7. [Multiple Symptoms]: If there are multiple keywords (e.g., back pain, headache) and no medication satisfies all, provide separate information for each symptom. (e.g., "For back pain, A is suitable; for headache, B is suitable.")
8. Verify that the search results (context) logically match the user's actual symptoms.
   - Example: If the user only says "click" but the search result is "arthritis", this is a wrong match, so do not recommend medication and ask them to rephrase the question.

[Output Format - Must follow the format below]

Ingredient: (ingredient name)
Efficacy: (efficacy)

Contraindication Warning: (if there are contraindicated medications in DUR information, must include)
- Medications that should not be taken with (ingredient name): (contraindicated ingredient) - (reason)

Suggestion: (suggestions related to usage)""",
        "JP": """あなたは韓国の医薬品情報専門AIチャットボットです。
韓国食品医薬品安全処のe약은요、医薬品承認情報データの内容のみに基づいて、
ユーザーの質問に正確かつ親切に答えます。

[必ず守るべきルール]
1. 検索して得た内容に基づいて成分名と効能をマッチングします。
2. ユーザーの症状に最も適した成分名を見つけます。
3. 該当成分名とその成分の代表的な効能を答えます。
4. 親切ですが防御的な表現で答えます。
5. ユーザーを「ユーザー」と呼びます。
6. 共感などの不要な表現はしません。
7. [複数症状の処理]: キーワードが複数（例：腰痛、頭痛）で、これらを同時に満たす薬がない場合は、それぞれの症状に合った薬情報を区別して答えてください。（例：「腰痛にはAが、頭痛にはBが適しています。」）
8. 検索結果（context）がユーザーの実際の症状と論理的に一致するか確認してください。
   - 例：ユーザーが「カチッ」とだけ言っているのに検索結果が「関節炎」の場合、これは誤ったマッチングなので、薬を推奨せず、質問を言い換えるよう依頼してください。

[出力形式 - 必ず以下の形式に従ってください]

成分名: (該当成分名)
効能: (該当効能)

併用禁忌注意: (DUR情報に併用禁忌薬物がある場合は必ず記載)
- (成分名)と併用してはいけない薬物: (併用禁忌成分名) - (理由)

提案: (服用に関する提案事項)"""
    }
    
    system_prompt = lang_instructions.get(language, lang_instructions["KR"])
    
    # Emergency & Context Filter 섹션을 언어별로 제공
    emergency_sections = {
        "KR": """
[Emergency & Context Filter]
사용자의 입력이 다음 중 하나에 해당할 경우, 응급 상황 출력 문구를 출력하십시오.

응급 상황(Emergency Keywords):

[외상 (Trauma)]
- 출혈: "피가 멈추지 않음", "대량 출혈", "피가 많이 난다", "동맥 출혈"
- 자상/찰과상: "칼에 깊이 베임", "유리에 찔림", "깊은 상처", "살이 보임"
- 골절: "부러짐", "뼈가 보임", "관절이 이상한 방향으로 꺾임"
- 절단: "손가락이 잘림", "신체 일부 절단", "절단 사고"
- 화상: "3도 화상", "피부가 검게 탐", "물집이 크게 생김", "화학 화상", "전기 화상"
- 두부 외상: "머리를 세게 부딪힘", "의식을 잃음", "구토가 나옴", "귀/코에서 피"

[호흡 (Respiratory)]
- 호흡곤란: "숨을 못 쉼", "숨이 막힘", "호흡이 힘듦", "기도 폐쇄"
- 질식: "목에 뭐가 걸림", "음식이 기도에 막힘", "숨을 쉴 수 없음"
- 천식 발작: "천식 발작", "인헬러로 안 됨", "입술이 파래짐"
- 익수: "물에 빠짐", "익사 직전", "물을 마심"

[순환 (Cardiovascular)]
- 심장: "가슴을 쥐어짜는 통증", "심장마비", "가슴이 찢어지는 듯함", "왼쪽 팔 저림"
- 뇌졸중: "얼굴 한쪽 마비", "말이 어눌함", "팔다리에 힘이 없음", "갑자기 시야가 흐림"
- 실신: "의식을 잃음", "쓰러짐", "깨어나지 않음"

[알레르기 (Allergy)]
- 아나필락시스: "목이 부어 숨을 못 쉼", "전신 두드러기", "얼굴/혀가 부음", "혈압이 급격히 떨어짐"
- 벌에 쏘임 후 쇼크: "벌에 쏘인 후 숨이 힘듦", "온몸이 붓고 두드러기"
- 음식 알레르기 쇼크: "음식 먹고 목이 막힘", "알레르기 쇼크"

[중독 (Poisoning)]
- 약물 과다복용: "약을 많이 먹음", "수면제 과다복용", "자살 시도"
- 화학물질: "세제를 마심", "농약 노출", "가스 흡입", "일산화탄소"
- 알코올: "술을 너무 많이 마심", "의식이 없음", "구토 후 질식"

[신경 (Neurological)]
- 경련/발작: "경련", "발작", "간질 발작", "몸이 떨림", "거품을 물음"
- 의식저하: "깨우지 못함", "반응이 없음", "혼수상태"
- 급성 두통: "벼락치는 듯한 두통", "인생 최악의 두통"

[환경 응급 (Environmental)]
- 온열 질환: "열사병", "체온이 40도 이상", "땀이 안 남"
- 저체온증: "체온이 떨어짐", "몸이 굳음", "입술이 파래짐"
- 감전: "전기에 감전", "번개에 맞음"
- 동상: "손발이 검게 변함", "감각이 없음"

[기타 응급]
- 임신 관련: "대량 출혈", "양수가 터짐", "태동이 없음"
- 소아: "영아 질식", "경련", "고열(40도 이상)"
- 정신과적 응급: "자해", "자살 시도", "타인 해칠 위험"

시점 판단 로직:
"절단 후", "수술 후", "상흔(흉터)" 등 '후(Post-)'의 상태인지, 아니면 '지금 막 발생한' 상황인지 구분하십시오.
제품 설명에 있는 '절단'은 '절단 수술이 완료되고 상처가 아문 후의 흉터 관리'를 의미합니다. 현재 사고 상황에는 절대 추천하지 마십시오.
응급 상황으로 판단될 때는 부연 설명 없이 응급 상황 출력 문구를 그대로 출력하세요.

응급 상황 출력 문구:
[Emergency Response]
- 현재 상황은 즉각적인 응급 처치와 병원 진료가 필요한 응급 상황으로 판단됩니다.
- 일반 의약품(연고 등)을 임의로 사용하면 감염이나 증상 악화의 위험이 있습니다.
- 즉시 119에 연락하거나 가까운 응급실로 방문하십시오.

"사용자가 언급한 키워드가 제품의 '적응증(Indication)'에 포함되어 있더라도, 그것이 '흉터(Scar)'나 '사후 관리(Post-care)' 목적이 아닌 '급성 외상(Acute Trauma)' 상황이라면 정보를 제공하지 마십시오."

[병용금지 안내 규칙 - 반드시 준수]
1. DUR(의약품안전사용서비스) 병용금지 정보가 제공되면, 반드시 "병용금지 주의:" 섹션을 작성하여 안내합니다.
2. 검색된 약품들 간에 상호 병용금지가 있으면 (예: "[검색된 약품 간 상호 병용금지 경고]"가 포함된 경우) 반드시 경고 메시지를 포함합니다.
3. 병용금지 사유(부작용 위험)를 반드시 함께 안내합니다.
4. 병용금지 정보가 "(병용금지 정보 없음)"인 경우에만 병용금지 섹션을 생략합니다.
""",
        "EN": """
[Emergency & Context Filter]
If the user's input corresponds to any of the following, output the emergency situation message.

Emergency Situations (Emergency Keywords):

[Trauma]
- Bleeding: "bleeding won't stop", "massive bleeding", "heavy bleeding", "arterial bleeding"
- Laceration/Abrasion: "deep cut with knife", "stabbed by glass", "deep wound", "flesh visible"
- Fracture: "broken", "bone visible", "joint bent in wrong direction"
- Amputation: "finger cut off", "body part severed", "amputation accident"
- Burn: "3rd degree burn", "skin blackened", "large blisters", "chemical burn", "electrical burn"
- Head Trauma: "hit head hard", "lost consciousness", "vomiting", "bleeding from ear/nose"

[Respiratory]
- Breathing Difficulty: "can't breathe", "choking", "difficulty breathing", "airway obstruction"
- Choking: "something stuck in throat", "food blocked airway", "can't breathe"
- Asthma Attack: "asthma attack", "inhaler not working", "lips turning blue"
- Drowning: "fell into water", "near drowning", "swallowed water"

[Cardiovascular]
- Heart: "crushing chest pain", "heart attack", "chest tearing pain", "left arm numbness"
- Stroke: "one side of face paralyzed", "slurred speech", "weakness in limbs", "sudden blurred vision"
- Fainting: "lost consciousness", "collapsed", "won't wake up"

[Allergy]
- Anaphylaxis: "throat swollen can't breathe", "full body hives", "face/tongue swollen", "blood pressure dropping rapidly"
- Bee Sting Shock: "stung by bee, difficulty breathing", "whole body swollen with hives"
- Food Allergy Shock: "ate food, throat blocked", "allergy shock"

[Poisoning]
- Drug Overdose: "took too much medicine", "sleeping pill overdose", "suicide attempt"
- Chemical: "drank detergent", "pesticide exposure", "gas inhalation", "carbon monoxide"
- Alcohol: "drank too much", "unconscious", "vomiting and choking"

[Neurological]
- Seizure/Convulsion: "seizure", "convulsion", "epileptic seizure", "body shaking", "foaming at mouth"
- Decreased Consciousness: "can't wake up", "no response", "coma"
- Acute Headache: "thunderclap headache", "worst headache of life"

[Environmental Emergency]
- Heat Illness: "heat stroke", "body temperature above 40°C", "no sweating"
- Hypothermia: "body temperature dropping", "body stiff", "lips blue"
- Electric Shock: "electrocuted", "struck by lightning"
- Frostbite: "hands/feet blackened", "no sensation"

[Other Emergency]
- Pregnancy Related: "massive bleeding", "water broke", "no fetal movement"
- Pediatric: "infant choking", "seizure", "high fever (above 40°C)"
- Psychiatric Emergency: "self-harm", "suicide attempt", "risk of harming others"

Timing Judgment Logic:
Distinguish whether it is a 'post-' state like "after amputation", "after surgery", "scar" or a 'just occurred' situation.
The 'amputation' in product description means 'scar management after amputation surgery is complete and wound has healed'. Never recommend for current accident situations.
When judged as an emergency situation, output the emergency situation message without additional explanation.

Emergency Situation Output Message:
[Emergency Response]
- The current situation is judged to be an emergency requiring immediate first aid and hospital treatment.
- Using general medications (ointments, etc.) arbitrarily may risk infection or symptom worsening.
- Immediately contact 119 or visit the nearest emergency room.

"Even if the keyword mentioned by the user is included in the product's 'Indication', if it is an 'Acute Trauma' situation rather than 'Scar' or 'Post-care' purpose, do not provide information."

[Contraindication Guidance Rules - Must Follow]
1. If DUR (Drug Utilization Review) contraindication information is provided, you must create a "Contraindication Warning:" section.
2. If there is mutual contraindication among searched medications (e.g., if "[Mutual Contraindication Warning Among Searched Medications]" is included), you must include a warning message.
3. You must also inform about the contraindication reason (side effect risk).
4. Only omit the contraindication section if the contraindication information is "(No contraindication information)".
""",
        "JP": """
[Emergency & Context Filter]
ユーザーの入力が以下のいずれかに該当する場合、緊急状況出力文を出力してください。

緊急状況（Emergency Keywords）:

[外傷 (Trauma)]
- 出血: "血が止まらない", "大量出血", "血がたくさん出る", "動脈出血"
- 切創/擦過傷: "ナイフで深く切れた", "ガラスに刺された", "深い傷", "肉が見える"
- 骨折: "折れた", "骨が見える", "関節が異常な方向に曲がっている"
- 切断: "指が切れた", "身体の一部が切断された", "切断事故"
- 火傷: "3度火傷", "皮膚が黒く焼けた", "大きな水ぶくれ", "化学火傷", "感電火傷"
- 頭部外傷: "頭を強く打った", "意識を失った", "嘔吐が出る", "耳/鼻から血"

[呼吸 (Respiratory)]
- 呼吸困難: "息ができない", "息が詰まる", "呼吸が困難", "気道閉塞"
- 窒息: "喉に何かが詰まった", "食べ物が気道に詰まった", "息ができない"
- 喘息発作: "喘息発作", "吸入器が効かない", "唇が青くなる"
- 溺水: "水に落ちた", "溺死寸前", "水を飲んだ"

[循環 (Cardiovascular)]
- 心臓: "胸を締め付ける痛み", "心臓発作", "胸が裂けるような痛み", "左腕のしびれ"
- 脳卒中: "顔の片側が麻痺", "言葉が不明瞭", "手足に力がない", "突然視界がぼやける"
- 失神: "意識を失った", "倒れた", "目が覚めない"

[アレルギー (Allergy)]
- アナフィラキシー: "喉が腫れて息ができない", "全身じんましん", "顔/舌が腫れる", "血圧が急激に下がる"
- 蜂刺されショック: "蜂に刺されて息が困難", "全身が腫れてじんましん"
- 食物アレルギーショック: "食べ物を食べて喉が詰まった", "アレルギーショック"

[中毒 (Poisoning)]
- 薬物過量摂取: "薬をたくさん飲んだ", "睡眠薬過量摂取", "自殺企図"
- 化学物質: "洗剤を飲んだ", "農薬暴露", "ガス吸入", "一酸化炭素"
- アルコール: "お酒を飲みすぎた", "意識がない", "嘔吐して窒息"

[神経 (Neurological)]
- けいれん/発作: "けいれん", "発作", "てんかん発作", "体が震える", "泡を吹く"
- 意識低下: "起こせない", "反応がない", "昏睡状態"
- 急性頭痛: "雷のような頭痛", "人生最悪の頭痛"

[環境緊急 (Environmental)]
- 熱中症: "熱射病", "体温が40度以上", "汗が出ない"
- 低体温症: "体温が下がる", "体が硬直", "唇が青くなる"
- 感電: "電気に感電", "雷に打たれた"
- 凍傷: "手足が黒くなる", "感覚がない"

[その他緊急]
- 妊娠関連: "大量出血", "羊水が破れた", "胎動がない"
- 小児: "乳児窒息", "けいれん", "高熱（40度以上）"
- 精神科的緊急: "自傷", "自殺企図", "他人を傷つける危険"

時点判断ロジック:
"切断後", "手術後", "瘢痕（傷跡）"など'後（Post-）'の状態か、それとも'今まさに発生した'状況かを区別してください。
製品説明にある'切断'は'切断手術が完了し傷が治った後の瘢痕管理'を意味します。現在の事故状況には絶対に推奨しないでください。
緊急状況と判断された場合は、補足説明なしで緊急状況出力文をそのまま出力してください。

緊急状況出力文:
[Emergency Response]
- 現在の状況は即座の応急処置と病院診療が必要な緊急状況と判断されます。
- 一般医薬品（軟膏など）を任意に使用すると感染や症状悪化のリスクがあります。
- すぐに119に連絡するか、最寄りの救急室を訪問してください。

"ユーザーが言及したキーワードが製品の'適応症（Indication）'に含まれていても、それが'瘢痕（Scar）'や'事後管理（Post-care）'目的ではなく'急性外傷（Acute Trauma）'状況であれば、情報を提供しないでください。"

[併用禁忌案内ルール - 必ず遵守]
1. DUR（医薬品安全使用サービス）併用禁忌情報が提供された場合、必ず"併用禁忌注意:"セクションを作成して案内します。
2. 検索された医薬品間で相互併用禁忌がある場合（例："[検索された医薬品間相互併用禁忌警告]"が含まれる場合）、必ず警告メッセージを含めます。
3. 併用禁忌理由（副作用リスク）を必ず一緒に案内します。
4. 併用禁忌情報が"（併用禁忌情報なし）"の場合のみ併用禁忌セクションを省略します。
"""
    }
    
    emergency_section = emergency_sections.get(language, emergency_sections["KR"])
    
    # 언어별 human 메시지 템플릿
    human_templates = {
        "KR": "질문: {question}\n\n검색 방식: {category} 컬럼에서 \"{keyword}\" 검색\n\n검색 결과:\n{context}\n\n병용금지 정보(DUR):\n{dur_context}",
        "EN": "Question: {question}\n\nSearch Method: Search \"{keyword}\" in {category} column\n\nSearch Results:\n{context}\n\nContraindication Information (DUR):\n{dur_context}",
        "JP": "質問: {question}\n\n検索方式: {category}カラムで「{keyword}」を検索\n\n検索結果:\n{context}\n\n併用禁忌情報(DUR):\n{dur_context}"
    }
    
    # 답변 언어 명시 지시 추가
    language_instruction = {
        "KR": "\n\n중요: 모든 답변은 반드시 한국어로 작성하세요.",
        "EN": "\n\nImportant: All responses must be written in English.",
        "JP": "\n\n重要: すべての回答は必ず日本語で記述してください。"
    }
    
    full_system_prompt = system_prompt + "\n\n" + emergency_section + language_instruction.get(language, "")
    human_template = human_templates.get(language, human_templates["KR"])
    
    return ChatPromptTemplate.from_messages([
        ("system", full_system_prompt),
        ("human", human_template),
    ])


# --- [언어 자동 감지 함수] ---
def detect_language(text: str) -> str:
    """
    입력 텍스트의 언어를 자동 감지합니다.
    KR: 한글, EN: 영어, JP: 일본어
    """
    if not text:
        return "KR"  # 기본값
    
    # 한글 유니코드 범위: AC00-D7AF (완성형 한글)
    korean_count = sum(1 for char in text if '\uAC00' <= char <= '\uD7AF')
    
    # 일본어 문자 범위
    # 히라가나: 3040-309F, 가타카나: 30A0-30FF, 한자: 4E00-9FAF
    japanese_count = sum(1 for char in text if 
                        ('\u3040' <= char <= '\u309F') or  # 히라가나
                        ('\u30A0' <= char <= '\u30FF') or  # 가타카나
                        ('\u4E00' <= char <= '\u9FAF'))   # 한자
    
    # 영어는 알파벳과 공백, 숫자, 특수문자로 구성
    # 한글/일본어가 없고 알파벳이 많으면 영어로 판단
    english_chars = sum(1 for char in text if char.isascii() and (char.isalpha() or char.isspace()))
    
    # 비율 계산
    total_chars = len([c for c in text if not c.isspace()])
    if total_chars == 0:
        return "KR"
    
    korean_ratio = korean_count / total_chars if total_chars > 0 else 0
    japanese_ratio = japanese_count / total_chars if total_chars > 0 else 0
    english_ratio = english_chars / total_chars if total_chars > 0 else 0
    
    # 한글이 가장 많으면 한국어
    if korean_count > 0 and korean_ratio > 0.1:
        return "KR"
    # 일본어 문자가 있으면 일본어
    elif japanese_count > 0 and japanese_ratio > 0.1:
        return "JP"
    # 영어 알파벳이 많으면 영어
    elif english_chars > 0 and english_ratio > 0.5:
        return "EN"
    # 기본값은 한국어
    else:
        return "KR"


# --- [수정 부분 2: 가공 함수 보강 - 다국어 지원] ---
def format_answer(text: str, language: str = "KR"):
    """
    텍스트에 이미 줄바꿈이 섞여 있어도 강제로 '성분명:' 앞에 
    빈 줄을 만들어주는 더 강력한 로직입니다.
    """
    if not text:
        return text
    
    texts = LANG_TEXTS[language]
    
    # 언어별 키워드 매핑
    keyword_map = {
        "KR": "성분명:",
        "EN": "Ingredient:",
        "JP": "成分名:"
    }
    
    keyword = keyword_map.get(language, "성분명:")
    
    # 1. 모든 키워드 앞에 줄바꿈 두 개(\n\n)를 넣습니다.
    text = text.replace(keyword, f"\n\n{keyword}")
    
    # 2. 맨 처음에 오는 키워드 때문에 생긴 맨 위의 빈 줄만 분석 결과로 변경.
    result_label = texts["analysis_result"]
    return f'{result_label}\n {text}'


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain_with_sources()
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False
if "language" not in st.session_state:
    st.session_state.language = "KR"
if "language_auto_detect" not in st.session_state:
    st.session_state.language_auto_detect = True  # 기본값: 자동 감지 활성화
if "manual_language" not in st.session_state:
    st.session_state.manual_language = None  # 수동 선택된 언어 (None이면 자동 감지)


# 면책동의 다이얼로그
@st.dialog(title="⚠️ 면책사항 동의", width="large")
def disclaimer_dialog():
    """첫 진입 시 표시되는 면책사항 동의 팝업"""
    st.markdown(
        """

# **서비스 이용 약관 및 법적 면책 고지**

### 본 서비스는 식품의약품안전처 공공데이터를 기반으로 정보를 제공하는 데이터 검색 보조 도구입니다.

### 사용자는 본 서비스를 이용함과 동시에 아래의 모든 사항에 **동의**한 것으로 간주됩니다.

---

## **1. 의료 행위의 부인**

### 본 시스템이 제공하는 모든 정보는 일반적인 정보 제공만을 목적으로 하며, 의학적 진단, 치료, 처방 또는 복약 지도를 대신할 수 없습니다.

> **AI가 생성한 답변을 근거로 스스로 질병을 진단하거나 약물을 선택하여 복용하지 마십시오. 이는 오남용으로 인한 심각한 부작용을 초래할 수 있습니다.**

## **2. 정보의 정확성 및 최신성 보장 불가**

### 본 서비스는 생성형 AI(RAG) 기술을 사용합니다.

### AI의 특성상 환각 현상(Hallucination)이 발생할 수 있으며, 공공데이터의 내용과 다른 부정확하거나 왜곡된 정보를 제공할 가능성이 항상 존재합니다.

> 데이터베이스 업데이트 지연으로 인해 최신 의약품 정보나 허가 취소 사항이 반영되지 않았을 수 있습니다.

> 정보의 최종 확인은 반드시 공식적인 식약처 의약품안전나라 또는 전문가를 통해 확인하시기 바랍니다.

## **3. 책임의 제한**

### 서비스 운영 주체는 본 서비스가 제공한 정보의 오류, 누락, 지연으로 인해 발생하는 어떠한 형태의 직접적·간접적·결과적 손해(신체적 부상, 질환의 악화, 경제적 손실 등)에 대해서도 법적 책임을 지지 않습니다.

> **사용자가 본 시스템의 정보를 신뢰하여 행한 모든 결정 및 행동에 대한 책임은 전적으로 사용자 본인에게 있습니다.**

## **4. 전문가 상담 필수**

### 증상이 있거나 의약품 성분에 대해 궁금한 점이 있을 경우, 반드시 전문의 또는 약사와 상담하십시오.

### 응급 상황이 발생한 경우, 본 시스템에 의존하지 말고 즉시 응급 의료 기관(119 등)에 연락하십시오.

## **5. 데이터 출처 및 오용 금지**

### 본 서비스는 식약처의 공공데이터를 인용하나, 식약처가 본 서비스의 운영이나 결과물을 보증하는 것은 아닙니다.

> 사용자는 본 서비스의 결과를 상업적으로 이용하거나, 타인에게 의학적 권고로 전달하여 발생하는 모든 법적 문제에 대해 단독으로 책임을 집니다.

---

## **6. 확인 및 동의**

본인은 위 면책 고지 사항을 충분히 숙지하였으며, 본 서비스가 제공하는 정보는 참고용일 뿐 의료 전문가의 조언을 대체할 수 없음에 동의합니다.

또한, 이를 어기고 발생한 모든 결과에 대해 서비스 제공자에게 책임을 묻지 않을 것을 서약합니다.
        """
    )
    
    # 체크박스 상태 확인
    checked = st.checkbox("**내용을 꼼꼼히 확인 했습니다.**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 동의합니다", type="primary", use_container_width=True, disabled=not checked):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    with col2:
        if st.button("❌ 거부합니다", use_container_width=True):
            # 거부 시 Google로 리다이렉트 (브라우저에서 window.close()는 제한적)
            st.markdown(
                """
                <meta http-equiv="refresh" content="0; url=https://www.google.com">
                <script>window.location.href = 'https://www.google.com';</script>
                """,
                unsafe_allow_html=True,
            )
            st.stop()


# 면책동의 확인 - 동의하지 않으면 팝업 표시 후 중단
if not st.session_state.disclaimer_accepted:
    disclaimer_dialog()
    st.stop()


# 클립보드 복사 버튼 생성 함수
def copy_button(text: str, button_text: str):
    """클릭 시 텍스트를 클립보드에 복사하는 버튼 생성"""
    html_code = f"""
    <button onclick="navigator.clipboard.writeText('{text}').then(() => {{
        this.innerHTML = '✅ 복사됨!';
        setTimeout(() => {{ this.innerHTML = '{button_text}'; }}, 1500);
    }})" style="
        padding: 8px 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        width: 100%;
        margin: 4px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    " onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='0 4px 12px rgba(102,126,234,0.4)';"
       onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
        {button_text}
    </button>
    """
    components.html(html_code, height=50)


# UI는 항상 한국어로 고정
texts = LANG_TEXTS["KR"]

# 사이드바
with st.sidebar:
    # --- [다국어 지원: 언어 선택 UI] ---
    st.markdown("### 🌐 언어 선택 / Language / 言語選択")
    
    # 자동 감지 토글
    auto_detect = st.checkbox("🔄 자동 감지", value=st.session_state.language_auto_detect, help="질문 입력 시 자동으로 언어를 감지합니다")
    st.session_state.language_auto_detect = auto_detect
    
    # 언어 선택 버튼
    if not auto_detect:
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("KR", use_container_width=True, type="primary" if st.session_state.manual_language == "KR" else "secondary"):
                st.session_state.manual_language = "KR"
                st.rerun()
        with col2:
            if st.button("EN", use_container_width=True, type="primary" if st.session_state.manual_language == "EN" else "secondary"):
                st.session_state.manual_language = "EN"
                st.rerun()
        with col3:
            if st.button("JP", use_container_width=True, type="primary" if st.session_state.manual_language == "JP" else "secondary"):
                st.session_state.manual_language = "JP"
                st.rerun()
    else:
        # 자동 감지 모드일 때는 manual_language 초기화
        st.session_state.manual_language = None
    
    st.divider()
    
    st.title(texts["sidebar_title"])
    st.text(texts["sidebar_guide"])

    st.text(texts["example_title"])
    copy_button("타이레놀의 효능은 무엇인가요?", texts["example1"])
    copy_button("아세트아미노펜이 포함된 약은?", texts["example2"])
    copy_button("두통에 효과있는 약은?", texts["example3"])
    
    st.caption(f"분류기: {CLASSIFIER_MODEL}")
    st.caption(f"답변 생성: {LLM_MODEL}")
    st.caption("데이터: 식품의약품안전처 e약은요 + 허가정보")
    st.warning(texts["warning"])
    if st.button(texts["reset"]):
        st.session_state.messages = []
        st.rerun()


# 메인 UI
st.title(texts["title"])
st.caption(texts["caption"])


# --- [수정 부분 3: 대화 기록 표시 시 format_answer 적용] ---
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''
            <div class="user-message-group">
                <div class="user-icon">👤</div>
                <div class="user-bubble-container">
                    <div class="chat-bubble">{message["content"]}</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            # 출력 전에 텍스트를 가공하여 간격을 벌립니다.
            message_lang = message.get("language", "KR")  # 저장된 언어 정보 사용, 없으면 기본값 KR
            formatted_content = format_answer(message["content"], message_lang)
            st.markdown(f'<div class="chat-bubble">{formatted_content}</div>', unsafe_allow_html=True)

            if message.get("dur_data"):
                with st.expander(texts["contraindication_title"], expanded=False):
                    for ingredient, contraindications in message["dur_data"].items():
                        st.markdown(f"**[{ingredient}]** {texts['contraindication_text']}")
                        seen_mixtures = set()
                        for item in contraindications:
                            mixture = item.get("MIXTURE_INGR_KOR_NAME") or item.get("mixture_ingr_kor_name", "")
                            reason = item.get("PROHBT_CONTENT") or item.get("prohbt_content", "")
                            if mixture and mixture not in seen_mixtures:
                                seen_mixtures.add(mixture)
                                st.markdown(f"- {mixture}: {reason}")
                        st.divider()

            if "sources" in message and message["sources"]:
                with st.expander(texts["sources_title"]):
                    for src in message["sources"]:
                        st.text(f"{src['item_name']} | {texts['company']}: {src['entp_name']} | {texts['code']}: {src['item_seq']}")


# --- [수정 부분 4: 채팅 입력 처리 시 format_answer 적용 - 다국어 지원] ---
if user_input := st.chat_input(texts["placeholder"]):
    # 1. 사용자 질문을 화면에 즉시 렌더링 (커스텀 CSS 적용된 버전)
    st.markdown(f'''
        <div class="user-message-group">
            <div class="user-icon">👤</div>
            <div class="user-bubble-container">
                <div class="chat-bubble">{user_input}</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # 2. 답변 생성용 언어 처리 (UI는 한국어로 고정)
    if st.session_state.language_auto_detect:
        # 자동 감지 모드: 사용자 입력 기반으로 언어 감지
        detected_lang = detect_language(user_input)
        answer_lang = detected_lang
    else:
        # 수동 선택 모드: 사이드바에서 선택한 언어 사용
        answer_lang = st.session_state.manual_language or "KR"  # 선택 안 된 경우 기본값 KR

    # 3. 어시스턴트 답변 생성 과정
    with st.chat_message("assistant"):
        with st.spinner(texts["searching"]):
            prepared = prepare_context(user_input)
            source_drugs = prepared["source_drugs"]
            
            # 답변 생성용 언어별 프롬프트로 교체 (UI는 한국어 고정)
            lang_prompt = get_answer_prompt_with_language(answer_lang)
            prepared["prompt_messages"] = lang_prompt.format_messages(
                question=prepared["question"],
                category=prepared["category"],
                keyword=prepared["keyword"],
                context=prepared["context"],
                dur_context=prepared.get("dur_context", ""),
            )

        answer_placeholder = st.empty()
        full_answer = ""

        for chunk in stream_answer(prepared):
            full_answer += chunk
            display_stream = format_answer(full_answer, answer_lang)
            # 스트리밍 중인 임시 답변 표시
            answer_placeholder.markdown(f'<div class="chat-bubble">{display_stream}▌</div>', unsafe_allow_html=True)
        
        # 스트리밍 완료 후 최종 답변 확정 표시
        final_answer = format_answer(full_answer, answer_lang)
        answer_placeholder.markdown(f'<div class="chat-bubble">{final_answer}</div>', unsafe_allow_html=True)

        if prepared.get("category") and prepared.get("keyword"):
            st.caption(f"{texts['search']}: {prepared['category']} → \"{prepared['keyword']}\"")

        # 병용금지 경고 UI
        dur_data = prepared.get("dur_data", {})

        # 각 성분별 병용금지 약물 목록
        if dur_data:
            with st.expander(texts["contraindication_title"], expanded=False):
                for ingredient, contraindications in dur_data.items():
                    st.markdown(f"**[{ingredient}]** {texts['contraindication_text']}")
                    seen_mixtures = set()
                    for item in contraindications:
                        mixture = item.get("MIXTURE_INGR_KOR_NAME") or item.get("mixture_ingr_kor_name", "")
                        reason = item.get("PROHBT_CONTENT") or item.get("prohbt_content", "")
                        if mixture and mixture not in seen_mixtures:
                            seen_mixtures.add(mixture)
                            st.markdown(f"- {mixture}: {reason}")
                    st.divider()

        # 소스 데이터 수집
        sources = []
        if source_drugs:
            with st.expander(texts["related_drugs"]):
                for drug in source_drugs:
                    source_info = {
                        "item_name": drug.get("item_name", ""),
                        "entp_name": drug.get("entp_name", ""),
                        "item_seq": drug.get("item_seq", ""),
                        "main_item_ingr": drug.get("main_item_ingr", "")
                    }
                    sources.append(source_info)
                    st.text(f"{source_info['item_name']} | {texts['company']}: {source_info['entp_name']}")

    # ---------------------------------------------------------
    # 3. ✨ [여기서부터 중요!] 모든 과정이 끝난 후 세션에 저장
    # ---------------------------------------------------------
    # (1) 사용자 질문 저장
    st.session_state.messages.append({"role": "user", "content": user_input})

    # (2) 어시스턴트 답변 저장 (이미 위에서 선언된 full_answer와 sources 사용)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
        "dur_data": dur_data,
        "language": answer_lang,  # 답변 생성에 사용된 언어 정보 저장
    })
    
    # (3) 화면 새로고침 (이걸 해야 회색 잔상이 사라지고 상단 for문이 깔끔하게 다시 그립니다)
    st.rerun()

