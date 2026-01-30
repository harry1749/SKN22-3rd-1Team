"""dur_list.json 데이터를 Supabase 'dur' 테이블로 업로드합니다."""

import json
import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client

# .env 파일 로드
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
DUR_JSON_PATH = Path(__file__).parent.parent / "data" / "raw" / "dur_list.json"
TABLE_NAME = "dur"
BATCH_SIZE = 500  # 한 번에 업로드할 레코드 수

# 테이블 생성 SQL
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dur (
    id BIGINT PRIMARY KEY,
    "TYPE_NAME" TEXT,
    "MIX_TYPE" TEXT,
    "INGR_CODE" TEXT,
    "INGR_ENG_NAME" TEXT,
    "INGR_KOR_NAME" TEXT,
    "MIX" TEXT,
    "ORI" TEXT,
    "CLASS" TEXT,
    "MIXTURE_MIX_TYPE" TEXT,
    "MIXTURE_INGR_CODE" TEXT,
    "MIXTURE_INGR_ENG_NAME" TEXT,
    "MIXTURE_INGR_KOR_NAME" TEXT,
    "MIXTURE_MIX" TEXT,
    "MIXTURE_ORI" TEXT,
    "MIXTURE_CLASS" TEXT,
    "NOTIFICATION_DATE" TIMESTAMPTZ,
    "PROHBT_CONTENT" TEXT,
    "REMARK" TEXT,
    "DEL_YN" BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 검색 성능을 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_dur_ingr_kor_name ON dur ("INGR_KOR_NAME");
CREATE INDEX IF NOT EXISTS idx_dur_mixture_ingr_kor_name ON dur ("MIXTURE_INGR_KOR_NAME");
CREATE INDEX IF NOT EXISTS idx_dur_del_yn ON dur ("DEL_YN");
"""


def create_table_if_not_exists(client) -> None:
    """dur 테이블이 없으면 생성합니다."""
    print("Checking/Creating 'dur' table...")
    try:
        client.postgrest.rpc("", {}).execute()  # 연결 테스트
    except:
        pass

    # SQL 실행 (Supabase SQL Editor에서 실행 필요)
    try:
        result = client.table(TABLE_NAME).select("id").limit(1).execute()
        print("Table 'dur' already exists")
    except Exception as e:
        if "does not exist" in str(e) or "relation" in str(e).lower():
            print("\n" + "=" * 50)
            print("ERROR: 'dur' table does not exist!")
            print("=" * 50)
            print("\nPlease run this SQL in Supabase SQL Editor:")
            print("-" * 50)
            print(CREATE_TABLE_SQL)
            print("-" * 50)
            print("\nAfter creating the table, run this script again.")
            sys.exit(1)
        else:
            raise


def load_dur_data() -> list[dict]:
    """dur_list.json 파일을 로드합니다."""
    print(f"Loading DUR data from: {DUR_JSON_PATH}")
    with open(DUR_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data


def upload_to_supabase(data: list[dict]) -> None:
    """데이터를 Supabase dur 테이블에 업로드합니다."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 배치 단위로 업로드
    total = len(data)
    uploaded = 0

    for i in range(0, total, BATCH_SIZE):
        batch = data[i : i + BATCH_SIZE]
        try:
            # upsert로 중복 시 업데이트
            result = client.table(TABLE_NAME).upsert(batch).execute()
            uploaded += len(batch)
            print(f"Uploaded {uploaded}/{total} records...")
        except Exception as e:
            print(f"Error uploading batch {i//BATCH_SIZE + 1}: {e}")
            raise

    print(f"Successfully uploaded {uploaded} records to '{TABLE_NAME}' table")


def main():
    print("=" * 50)
    print("DUR Data Upload to Supabase")
    print("=" * 50)

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 테이블 존재 여부 확인
    create_table_if_not_exists(client)

    # 데이터 로드
    data = load_dur_data()

    if not data:
        print("No data to upload")
        return

    # 업로드
    upload_to_supabase(data)

    print("=" * 50)
    print("Upload complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
