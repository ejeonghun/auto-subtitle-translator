#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT 자막 파일 Gemini API 번역기
작성자: AI Assistant
버전: 1.0
설명: SRT 자막 파일을 Gemini API를 사용하여 번역하는 프로그램
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
import sys

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai 라이브러리가 설치되지 않았습니다.")
    print("다음 명령어로 설치해주세요: pip install google-generativeai")
    sys.exit(1)

@dataclass
class SubtitleEntry:
    """자막 항목을 저장하는 데이터 클래스"""
    index: int
    start_time: str
    end_time: str
    text: str

class SRTTranslator:
    """SRT 파일을 Gemini API로 번역하는 클래스"""

    def __init__(self, api_key: str, target_language: str = "Korean", batch_size: int = 50):
        """
        초기화 함수

        Args:
            api_key: Gemini API 키
            target_language: 번역할 대상 언어
            batch_size: 한 번에 처리할 자막 개수 (토큰 제한 고려)
        """
        self.api_key = api_key
        # self.api_key = "AIzaSyBHmyFhjNDRitiC7CMHg-OTVJdzRmxe9tw"
        self.target_language = target_language
        self.batch_size = batch_size

        # Gemini API 설정
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("✅ Gemini API 연결 성공!")
        except Exception as e:
            print(f"❌ Gemini API 연결 실패: {e}")
            raise

    def parse_srt_file(self, file_path: str) -> List[SubtitleEntry]:
        """
        SRT 파일을 파싱하여 SubtitleEntry 리스트로 변환

        Args:
            file_path: SRT 파일 경로

        Returns:
            SubtitleEntry 리스트
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
        content = None

        # 여러 인코딩으로 파일 읽기 시도
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"✅ 파일 읽기 성공 (인코딩: {encoding})")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError("파일을 읽을 수 없습니다. 인코딩 문제일 수 있습니다.")

        # SRT 파일의 각 항목을 정규표현식으로 파싱
        # 개행 문자 정규화
        content = content.replace('\r\n', '\n').replace('\r', '\n')

        # SRT 패턴 매칭 (더 유연한 패턴)
        pattern = r'(\d+)\s*\n([\d:,]+)\s*-->\s*([\d:,]+)\s*\n([\s\S]*?)(?=\n\s*\d+\s*\n|\Z)'
        matches = re.findall(pattern, content, re.MULTILINE)

        subtitles = []
        for match in matches:
            try:
                index = int(match[0])
                start_time = match[1].strip()
                end_time = match[2].strip()
                text = match[3].strip().replace('\n', ' ').replace('  ', ' ')

                # 빈 텍스트는 건너뛰기
                if not text:
                    continue

                subtitles.append(SubtitleEntry(
                    index=index,
                    start_time=start_time,
                    end_time=end_time,
                    text=text
                ))
            except ValueError as e:
                print(f"⚠️ 자막 항목 파싱 오류 (인덱스 {match[0]}): {e}")
                continue

        return subtitles

    def create_translation_prompt(self, subtitles_batch: List[SubtitleEntry], program_name: str = None) -> str:
        """
        번역을 위한 프롬프트 생성

        Args:
            subtitles_batch: 번역할 자막 배치
            program_name: 방송 프로그램명 (선택사항)

        Returns:
            번역 프롬프트 문자열
        """
        # 자막 텍스트만 추출하여 번호를 매김
        numbered_subtitles = []
        for i, subtitle in enumerate(subtitles_batch, 1):
            numbered_subtitles.append(f"{i}. {subtitle.text}")

        subtitle_text = "\n".join(numbered_subtitles)

        program_info = f"방송 프로그램: {program_name}\n" if program_name else "방송 프로그램: 자동으로 Gemini가 찾아서 어떠한 프로그램인지를 알아주세요\n"

        prompt = f"""
{program_info}내용: 자동으로 Gemini가 찾아서 어떠한 프로그램인지를 알아주세요
프롬프트: "{self.target_language}"으로 자연스럽게 번역 해 주세요.

자막:
{subtitle_text}

번역 규칙:
1. 각 번호에 해당하는 자막을 "{self.target_language}"으로 번역해주세요.
2. 번역 결과는 "번호. 번역된 내용" 형식으로 제공해주세요.
3. 자막의 맥락과 뉘앙스를 고려하여 자연스럽게 번역해주세요.
4. 고유명사나 전문용어는 적절히 현지화해주세요.
5. 숫자 순서는 반드시 유지해주세요.
6. 번역이 불가능한 경우 원문을 그대로 유지해주세요.

번역 결과:
"""

        return prompt

    def parse_translation_response(self, response_text: str, original_batch: List[SubtitleEntry]) -> List[SubtitleEntry]:
        """
        Gemini 응답을 파싱하여 번역된 SubtitleEntry 리스트로 변환

        Args:
            response_text: Gemini의 응답 텍스트
            original_batch: 원본 자막 배치

        Returns:
            번역된 SubtitleEntry 리스트
        """
        translated_subtitles = []

        # 번역 결과에서 번호가 있는 줄들을 찾기
        lines = response_text.strip().split('\n')
        translated_texts = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # "번호. 내용" 형식 찾기 (더 유연한 패턴)
            match = re.match(r'^(\d+)\.\s*(.+)', line)
            if match:
                num = int(match.group(1))
                text = match.group(2).strip()
                translated_texts[num] = text

        # 원본 자막과 매칭하여 번역된 자막 생성
        for i, original in enumerate(original_batch):
            translated_text = translated_texts.get(i + 1, original.text)  # 번역 실패시 원본 사용

            print(f"🔄 번역된 자막: {translated_text}")

            translated_subtitles.append(SubtitleEntry(
                index=original.index,
                start_time=original.start_time,
                end_time=original.end_time,
                text=translated_text
            ))

        return translated_subtitles

    def translate_subtitles(self, subtitles: List[SubtitleEntry], program_name: str = None) -> List[SubtitleEntry]:
        """
        자막 리스트를 배치 단위로 번역

        Args:
            subtitles: 번역할 자막 리스트
            program_name: 방송 프로그램명 (선택사항)

        Returns:
            번역된 자막 리스트
        """
        translated_subtitles = []
        total_batches = (len(subtitles) + self.batch_size - 1) // self.batch_size

        print(f"\n📝 총 {len(subtitles)}개 자막을 {total_batches}개 배치로 나누어 번역합니다...")
        print(f"🎯 대상 언어: {self.target_language}")
        print(f"📦 배치 크기: {self.batch_size}")

        success_count = 0
        error_count = 0

        for i in range(0, len(subtitles), self.batch_size):
            batch = subtitles[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            print(f"\n🔄 배치 {batch_num}/{total_batches} 번역 중... ({len(batch)}개 자막)")

            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    # 번역 프롬프트 생성
                    prompt = self.create_translation_prompt(batch, program_name)

                    # 토큰 수 체크 (대략적)
                    token_estimate = len(prompt.split())
                    if token_estimate > 30000:  # Gemini의 토큰 제한 고려
                        print(f"⚠️ 토큰수가 많습니다 ({token_estimate}). 배치 크기를 줄이는 것을 권장합니다.")

                    # Gemini API 호출
                    response = self.model.generate_content(prompt)

                    if not response.text:
                        raise ValueError("빈 응답을 받았습니다.")

                    # 응답 파싱
                    translated_batch = self.parse_translation_response(response.text, batch)
                    translated_subtitles.extend(translated_batch)

                    print(f"✅ 배치 {batch_num} 번역 완료!")
                    success_count += len(batch)
                    break

                except Exception as e:
                    retry_count += 1
                    print(f"❌ 배치 {batch_num} 번역 오류 (시도 {retry_count}/{max_retries}): {e}")

                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # 지수 백오프
                        print(f"⏳ {wait_time}초 후 재시도...")
                        time.sleep(wait_time)
                    else:
                        print("❌ 최대 재시도 횟수 초과. 원본 텍스트를 유지합니다.")
                        translated_subtitles.extend(batch)
                        error_count += len(batch)

            # API 요청 제한을 위한 지연 (무료 티어: 15 RPM)
            if batch_num < total_batches:
                time.sleep(4)  # 안전한 간격

        print(f"\n📊 번역 완료: 성공 {success_count}개, 오류 {error_count}개")
        return translated_subtitles

    def save_srt_file(self, subtitles: List[SubtitleEntry], output_path: str):
        """
        번역된 자막을 SRT 파일로 저장

        Args:
            subtitles: 저장할 자막 리스트
            output_path: 출력 파일 경로
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as file:
                for subtitle in subtitles:
                    file.write(f"{subtitle.index}\n")
                    file.write(f"{subtitle.start_time} --> {subtitle.end_time}\n")
                    file.write(f"{subtitle.text}\n\n")

            print(f"✅ 번역된 SRT 파일이 저장되었습니다: {output_path}")

        except Exception as e:
            print(f"❌ 파일 저장 오류: {e}")
            raise

    def translate_srt_file(self, input_path: str, output_path: str, program_name: str = None):
        """
        SRT 파일을 번역하는 메인 함수

        Args:
            input_path: 입력 SRT 파일 경로
            output_path: 출력 SRT 파일 경로
            program_name: 방송 프로그램명 (선택사항)
        """
        print("🚀 SRT 파일 번역을 시작합니다...")
        print(f"📁 입력 파일: {input_path}")
        print(f"📁 출력 파일: {output_path}")
        print(f"🌍 대상 언어: {self.target_language}")
        print(f"📦 배치 크기: {self.batch_size}")

        start_time = time.time()

        try:
            # SRT 파일 파싱
            print("\n1️⃣ SRT 파일 파싱 중...")
            subtitles = self.parse_srt_file(input_path)
            print(f"✅ 총 {len(subtitles)}개의 자막을 발견했습니다.")

            if len(subtitles) == 0:
                print("❌ 유효한 자막을 찾을 수 없습니다.")
                return

            # 자막 번역
            print("\n2️⃣ 자막 번역 중...")
            translated_subtitles = self.translate_subtitles(subtitles, program_name)

            # 번역된 SRT 파일 저장
            print("\n3️⃣ 번역된 SRT 파일 저장 중...")
            self.save_srt_file(translated_subtitles, output_path)

            end_time = time.time()
            duration = end_time - start_time

            print(f"\n🎉 번역 완료! (소요시간: {duration:.1f}초)")
            print(f"📊 처리된 자막: {len(translated_subtitles)}개")

        except Exception as e:
            print(f"❌ 번역 중 오류 발생: {e}")
            raise

def main():
    """메인 함수 - 명령행 인터페이스"""

    parser = argparse.ArgumentParser(description='SRT 자막 파일을 Gemini API로 번역합니다.')
    parser.add_argument('input_file', help='입력 SRT 파일 경로')
    parser.add_argument('-o', '--output', help='출력 SRT 파일 경로 (기본값: input_translated.srt)')
    parser.add_argument('-k', '--api-key', help='Gemini API 키 (환경변수 GEMINI_API_KEY도 사용 가능)')
    parser.add_argument('-l', '--language', default='Korean', help='번역할 언어 (기본값: Korean)')
    parser.add_argument('-b', '--batch-size', type=int, default=50, help='배치 크기 (기본값: 50)')
    parser.add_argument('-p', '--program', help='방송 프로그램명')

    args = parser.parse_args()

    # API 키 확인
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Gemini API 키가 필요합니다.")
        print("다음 중 하나의 방법으로 API 키를 제공해주세요:")
        print("1. 명령행 옵션: -k YOUR_API_KEY")
        print("2. 환경변수: export GEMINI_API_KEY=YOUR_API_KEY")
        sys.exit(1)

    # 출력 파일 경로 설정
    if not args.output:
        input_name = os.path.splitext(args.input_file)[0]
        args.output = f"{input_name}_translated.srt"

    # 배치 크기 검증
    if args.batch_size < 1 or args.batch_size > 100:
        print("❌ 배치 크기는 1-100 사이여야 합니다.")
        sys.exit(1)

    try:
        # 번역기 초기화
        translator = SRTTranslator(
            api_key=api_key,
            target_language=args.language,
            batch_size=args.batch_size
        )

        # SRT 파일 번역
        translator.translate_srt_file(args.input_file, args.output, args.program)

    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
