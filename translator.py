#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRT 자막 파일 Gemini API 번역기 모듈
작성자: AI Assistant
버전: 2.0
설명: SRT 자막 파일을 Gemini API를 사용하여 번역하는 모듈
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai 라이브러리가 설치되지 않았습니다.")
    genai = None

@dataclass
class SubtitleEntry:
    """자막 항목을 저장하는 데이터 클래스"""
    index: int
    start_time: str
    end_time: str
    text: str

class SubtitleTranslator:
    """SRT 파일을 Gemini API로 번역하는 클래스"""

    def __init__(self, api_key: str, target_language: str = "Korean", batch_size: int = 100, program_name: Optional[str] = None, additional_context: Optional[str] = None):
        """
        초기화 함수

        Args:
            api_key: Gemini API 키
            target_language: 번역할 대상 언어
            batch_size: 한 번에 처리할 자막 개수 (토큰 제한 고려)
        """
        if genai is None:
            raise ImportError("google-generativeai 라이브러리가 설치되지 않았습니다.")
        
        self.api_key = api_key
        self.target_language = target_language
        self.batch_size = batch_size
        self.program_name = program_name
        self.additional_context = additional_context

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
        file_path = Path(file_path)
        if not file_path.exists():
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

    def create_translation_prompt(self, subtitles_batch: List[SubtitleEntry], program_name: Optional[str] = None, additional_context: Optional[str] = None) -> str:
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

        program_info = f"방송 프로그램: {program_name}\n" if program_name else "방송 프로그램: (불명확할 경우 맥락으로 추정)\n"
        extra_context = f"부연설명: {additional_context}\n" if additional_context else "부연설명: (없음)\n"

        prompt = f"""
역할: 당신은 방송/영상 콘텐츠 자막 전문 번역가입니다. 의미 보존과 자연스러움, 시청 흐름을 최우선으로 합니다.
{program_info}{extra_context}
대상 언어: {self.target_language}

자막(번호. 텍스트):
{subtitle_text}

요청 사항:
1) 각 번호에 해당하는 자막을 자연스럽게 {self.target_language}로 번역하세요.
2) 서로 연속된 두 자막의 타임스탬프 간격이 매우 짧고(대략 0.3~0.5초 이내) 문장 상 이어져야 자연스러운 경우, 두 자막을 하나의 문장으로 "자연스럽게" 붙여 번역하세요.
3) 존칭/구어체/일상 회화의 톤은 맥락에 맞게 유지하세요.
4) 고유명사/전문용어는 현지화하되 의미 손실이 없도록 주의하세요.
5) 출력 형식은 반드시 "번호. 번역문"으로 하세요. 병합한 경우에도 첫 번째 번호만 사용하여 "번호. 번역문"으로 출력하세요.
6) 판단이 애매하면 과도한 병합은 피하고 원문 흐름을 존중하세요.

출력:
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

    def merge_adjacent_subtitles(self, subtitles: List[SubtitleEntry], max_gap_seconds: float = 0.4) -> List[SubtitleEntry]:
        """
        인접 자막을 시간 간격이 짧고 문장적으로 이어질 때 병합합니다.

        병합 휴리스틱:
        - 두 자막의 시작-끝 간격이 max_gap_seconds 이하
        - 앞 자막이 강한 종결부호로 끝나지 않음 (., !, ?, …, ~, ♪ 등)

        Returns:
            병합된 자막 리스트
        """
        if not subtitles:
            return []

        def parse_time(ts: str) -> float:
            h, m, rest = ts.split(':')
            s, ms = rest.split(',')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        def fmt_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        merged: List[SubtitleEntry] = []
        buffer = subtitles[0]
        end_punctuations = set(['.', '!', '?', '…', '~', '♪', '"', '\''])

        for cur in subtitles[1:]:
            prev_end = parse_time(buffer.end_time)
            cur_start = parse_time(cur.start_time)
            gap = max(cur_start - prev_end, 0.0)

            prev_text = buffer.text.strip()
            ends_with_terminal = len(prev_text) > 0 and prev_text[-1] in end_punctuations

            if gap <= max_gap_seconds and not ends_with_terminal:
                # 병합
                combined_text = (prev_text + ' ' + cur.text.strip()).strip()
                buffer = SubtitleEntry(
                    index=buffer.index,
                    start_time=buffer.start_time,
                    end_time=cur.end_time,
                    text=combined_text
                )
            else:
                merged.append(buffer)
                buffer = cur

        merged.append(buffer)
        # 인덱스를 1..N 재부여 (SRT 규격)
        for i, sub in enumerate(merged, start=1):
            sub.index = i
        return merged

    def translate_subtitles(self, subtitles: List[SubtitleEntry], program_name: str = None, stream_callback: Optional[Any] = None) -> List[SubtitleEntry]:
        """
        자막 리스트를 배치 단위로 번역

        Args:
            subtitles: 번역할 자막 리스트
            program_name: 방송 프로그램명 (선택사항)

        Returns:
            번역된 자막 리스트
        """
        # 0) 번역 전 병합
        subtitles = self.merge_adjacent_subtitles(subtitles)

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
                    prompt = self.create_translation_prompt(
                        batch,
                        program_name or self.program_name,
                        self.additional_context
                    )

                    # 토큰 수 체크 (대략적)
                    token_estimate = len(prompt.split())
                    if token_estimate > 30000:  # Gemini의 토큰 제한 고려
                        print(f"⚠️ 토큰수가 많습니다 ({token_estimate}). 배치 크기를 줄이는 것을 권장합니다.")

                    # Gemini API 호출 (스트리밍 지원)
                    aggregated_text = ""
                    try:
                        response = self.model.generate_content(prompt, stream=True)
                        for chunk in response:
                            if hasattr(chunk, 'text') and chunk.text:
                                aggregated_text += chunk.text
                                if stream_callback:
                                    try:
                                        stream_callback(chunk.text)
                                    except Exception:
                                        pass
                        # ensure complete
                        try:
                            response.resolve()
                        except Exception:
                            pass
                    except TypeError:
                        # 라이브러리 버전에 따라 stream 미지원 시 폴백
                        response = self.model.generate_content(prompt)
                        aggregated_text = response.text or ""

                    if not aggregated_text.strip():
                        raise ValueError("빈 응답을 받았습니다.")

                    # 응답 파싱
                    translated_batch = self.parse_translation_response(aggregated_text, batch)
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
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as file:
                for subtitle in subtitles:
                    file.write(f"{subtitle.index}\n")
                    file.write(f"{subtitle.start_time} --> {subtitle.end_time}\n")
                    file.write(f"{subtitle.text}\n\n")

            print(f"✅ 번역된 SRT 파일이 저장되었습니다: {output_path}")

        except Exception as e:
            print(f"❌ 파일 저장 오류: {e}")
            raise

    def translate_srt_file(self, input_path: str, output_path: str, program_name: str = None, stream_callback: Optional[Any] = None) -> bool:
        """
        SRT 파일을 번역하는 메인 함수

        Args:
            input_path: 입력 SRT 파일 경로
            output_path: 출력 SRT 파일 경로
            program_name: 방송 프로그램명 (선택사항)
            
        Returns:
            번역 성공 여부
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
                return False

            # 자막 번역
            print("\n2️⃣ 자막 번역 중...")
            translated_subtitles = self.translate_subtitles(subtitles, program_name, stream_callback)

            # 번역된 SRT 파일 저장
            print("\n3️⃣ 번역된 SRT 파일 저장 중...")
            self.save_srt_file(translated_subtitles, output_path)

            end_time = time.time()
            duration = end_time - start_time

            print(f"\n🎉 번역 완료! (소요시간: {duration:.1f}초)")
            print(f"📊 처리된 자막: {len(translated_subtitles)}개")
            
            return True

        except Exception as e:
            print(f"❌ 번역 중 오류 발생: {e}")
            return False

    def close(self):
        """API 클라이언트 레퍼런스를 정리합니다."""
        try:
            if hasattr(self, "model"):
                self.model = None
        except Exception:
            pass

# main.py와의 호환성을 위한 래퍼 클래스
class Translator:
    """main.py와 호환되는 번역기 래퍼 클래스"""
    
    def __init__(self, api_key: str):
        """
        초기화 함수
        
        Args:
            api_key: Gemini API 키
        """
        self.api_key = api_key
        self._translator = None
    
    def translate_srt_file(self, input_path: str, output_path: str, target_language: str = "Korean") -> bool:
        """
        SRT 파일을 번역하는 함수
        
        Args:
            input_path: 입력 SRT 파일 경로
            output_path: 출력 SRT 파일 경로
            target_language: 번역할 대상 언어
            
        Returns:
            번역 성공 여부
        """
        try:
            # 대상 언어가 바뀔 때마다 새로운 SubtitleTranslator 인스턴스 생성
            if self._translator is None or self._translator.target_language != target_language:
                self._translator = SubtitleTranslator(
                    api_key=self.api_key,
                    target_language=target_language,
                    batch_size=50
                )
            
            return self._translator.translate_srt_file(input_path, output_path)
            
        except Exception as e:
            print(f"❌ 번역 중 오류 발생: {e}")
            return False

def main():
    """테스트용 메인 함수"""
    # 사용 예시
    # translator = SubtitleTranslator("YOUR_API_KEY", "Korean")
    # success = translator.translate_srt_file("input.srt", "output.srt")
    # if success:
    #     print("번역 완료!")
    pass

if __name__ == "__main__":
    main()