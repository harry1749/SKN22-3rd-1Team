from src.config import SEARCH_LIMIT, SUPABASE_KEY, SUPABASE_URL
from supabase import create_client

# 분류 카테고리 → Supabase drugs 테이블 컬럼 매핑
CATEGORY_COLUMN_MAP = {
    "product_name": "item_name",
    "ingredient": "main_item_ingr",
    "efficacy": "efcy_qesitm",
}

# 행 데이터를 텍스트로 변환할 때 사용할 필드 라벨
FIELD_LABELS = {
    "item_name": "제품명",
    "entp_name": "업체명",
    "item_seq": "품목기준코드",
    "main_item_ingr": "주성분",
    "chart": "성상",
    "spclty_pblc": "전문/일반",
    "item_permit_date": "허가일자",
    "efcy_qesitm": "효능",
    "use_method_qesitm": "사용법",
    "atpn_warn_qesitm": "주의사항 경고",
    "atpn_qesitm": "주의사항",
    "intrc_qesitm": "상호작용",
    "se_qesitm": "부작용",
    "deposit_method_qesitm": "보관법",
    "storage_method": "저장방법",
    "valid_term": "유효기간",
}


def _get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def search_drugs(category: str, keyword: str) -> list[dict]:
    """drugs 테이블에서 category에 해당하는 컬럼을 keyword로 ILIKE 검색합니다."""
    column = CATEGORY_COLUMN_MAP.get(category)
    if not column:
        return []

    client = _get_client()
    res = (
        client.table("drugs")
        .select("*")
        .ilike(column, f"%{keyword}%")
        .limit(SEARCH_LIMIT)
        .execute()
    )
    return res.data or []


def format_drug_info(row: dict) -> str:
    """drugs 테이블의 행 1건을 읽기 좋은 텍스트로 포맷합니다."""
    lines = []
    for key, label in FIELD_LABELS.items():
        value = (row.get(key) or "").strip()
        if value:
            lines.append(f"[{label}] {value}")
    return "\n".join(lines)


def format_search_results(rows: list[dict]) -> str:
    """검색 결과 전체를 하나의 컨텍스트 문자열로 포맷합니다."""
    if not rows:
        return "(검색 결과 없음)"
    parts = []
    for i, row in enumerate(rows, 1):
        header = f"── 검색 결과 {i} ──"
        body = format_drug_info(row)
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


# ── DUR 병용금지 검색 함수들 ──────────────────────────────────


def extract_ingredients(drugs_data: list[dict]) -> list[str]:
    """검색된 약품 데이터에서 성분명 목록을 추출합니다.

    drugs 테이블의 main_item_ingr 형식: "[M040548]창출|[M040486]육두구|..."
    - 파이프(|)로 구분
    - [코드] 접두사 포함
    """
    import re

    ingredients = set()
    for drug in drugs_data:
        main_ingr = drug.get("main_item_ingr", "")
        if main_ingr:
            # 파이프(|)로 구분된 성분들 처리
            for ingr in main_ingr.split("|"):
                cleaned = ingr.strip()
                # [코드] 형식 제거 (예: "[M040548]창출" → "창출")
                if cleaned.startswith("[") and "]" in cleaned:
                    cleaned = cleaned.split("]", 1)[1].strip()
                # 괄호 안 함량 정보 제거 (예: "아세트아미노펜(500mg)" → "아세트아미노펜")
                if "(" in cleaned:
                    cleaned = cleaned.split("(")[0].strip()
                if cleaned:
                    ingredients.add(cleaned)
    return list(ingredients)


def _normalize_ingredient_name(name: str) -> str:
    """성분명에서 접미사를 제거하여 핵심 이름만 추출합니다.

    예: "슈도에페드린염산염" → "슈도에페드린"
        "겐타마이신황산염" → "겐타마이신"
    """
    # 일반적인 제약 접미사 목록 (긴 것부터 처리)
    suffixes = [
        "염산염수화물", "브롬화수소산염수화물", "오로트산염수화물",
        "염산염", "황산염", "수화물", "말레산염", "푸마르산염",
        "타르타르산염", "인산염", "질산염", "아세트산염",
    ]
    normalized = name
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def search_dur_by_ingredient(ingredient_name: str) -> list[dict]:
    """성분명으로 dur 테이블에서 병용금지 약물을 검색합니다."""
    client = _get_client()

    # 원본 성분명으로 검색
    res = (
        client.table("dur")
        .select("*")
        .ilike("INGR_KOR_NAME", f"%{ingredient_name}%")
        .eq("DEL_YN", False)
        .limit(20)
        .execute()
    )

    results = res.data or []

    # 결과가 없으면 정규화된 이름으로 재검색
    if not results:
        normalized = _normalize_ingredient_name(ingredient_name)
        if normalized != ingredient_name:
            res2 = (
                client.table("dur")
                .select("*")
                .ilike("INGR_KOR_NAME", f"%{normalized}%")
                .eq("DEL_YN", False)
                .limit(20)
                .execute()
            )
            results = res2.data or []

    return results


def search_dur_for_ingredients(ingredients: list[str]) -> dict[str, list[dict]]:
    """여러 성분에 대해 각각 DUR 병용금지 정보를 검색합니다."""
    result = {}
    for ingr in ingredients:
        dur_data = search_dur_by_ingredient(ingr)
        if dur_data:
            result[ingr] = dur_data
    return result


def _get_dur_field(row: dict, field: str) -> str:
    """DUR 데이터에서 필드 값을 대소문자 구분 없이 가져옵니다."""
    return row.get(field) or row.get(field.lower(), "")


def check_mutual_contraindication(ingredients: list[str]) -> list[dict]:
    """검색된 약품들의 성분 간 상호 병용금지를 체크합니다."""
    if len(ingredients) < 2:
        return []

    mutual_warnings = []
    client = _get_client()

    # 성분 쌍을 순회하며 병용금지 관계 확인
    for i, ingr1 in enumerate(ingredients):
        for ingr2 in ingredients[i + 1 :]:
            # ingr1 → ingr2 방향 체크
            res = (
                client.table("dur")
                .select("*")
                .ilike("INGR_KOR_NAME", f"%{ingr1}%")
                .ilike("MIXTURE_INGR_KOR_NAME", f"%{ingr2}%")
                .eq("DEL_YN", False)
                .limit(5)
                .execute()
            )
            if res.data:
                for row in res.data:
                    mutual_warnings.append(
                        {
                            "drug1": _get_dur_field(row, "INGR_KOR_NAME"),
                            "drug2": _get_dur_field(row, "MIXTURE_INGR_KOR_NAME"),
                            "reason": _get_dur_field(row, "PROHBT_CONTENT"),
                        }
                    )
            # ingr2 → ingr1 방향 체크 (역방향)
            res2 = (
                client.table("dur")
                .select("*")
                .ilike("INGR_KOR_NAME", f"%{ingr2}%")
                .ilike("MIXTURE_INGR_KOR_NAME", f"%{ingr1}%")
                .eq("DEL_YN", False)
                .limit(5)
                .execute()
            )
            if res2.data:
                for row in res2.data:
                    mutual_warnings.append(
                        {
                            "drug1": _get_dur_field(row, "INGR_KOR_NAME"),
                            "drug2": _get_dur_field(row, "MIXTURE_INGR_KOR_NAME"),
                            "reason": _get_dur_field(row, "PROHBT_CONTENT"),
                        }
                    )

    return mutual_warnings


def format_dur_results(dur_data: dict[str, list[dict]]) -> str:
    """DUR 검색 결과를 LLM 컨텍스트용 텍스트로 포맷합니다."""
    if not dur_data:
        return "(병용금지 정보 없음)"

    parts = []
    for ingredient, contraindications in dur_data.items():
        lines = [f"[{ingredient}의 병용금지 약물]"]
        seen = set()
        for item in contraindications:
            # 대소문자 모두 처리 (Supabase 컬럼명 호환)
            mixture = item.get("MIXTURE_INGR_KOR_NAME") or item.get("mixture_ingr_kor_name", "")
            reason = item.get("PROHBT_CONTENT") or item.get("prohbt_content", "")
            if mixture and mixture not in seen:
                seen.add(mixture)
                lines.append(f"- {mixture}: {reason}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def format_mutual_warnings(mutual_warnings: list[dict]) -> str:
    """상호 병용금지 경고를 LLM 컨텍스트용 텍스트로 포맷합니다."""
    if not mutual_warnings:
        return ""

    lines = ["[검색된 약품 간 상호 병용금지 경고]"]
    seen = set()
    for warn in mutual_warnings:
        key = (warn["drug1"], warn["drug2"])
        if key not in seen:
            seen.add(key)
            lines.append(f"- {warn['drug1']} + {warn['drug2']}: {warn['reason']}")

    return "\n".join(lines)
