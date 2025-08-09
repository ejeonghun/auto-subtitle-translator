#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any
from threading import Thread, Event
import time

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                            QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
                            QPushButton, QLabel, QLineEdit, QTextEdit, QPlainTextEdit,
                            QProgressBar, QComboBox, QListWidget, QListWidgetItem,
                            QFileDialog, QMessageBox, QCheckBox, QFrame, QSplitter,
                            QScrollArea, QGroupBox)
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtCore import Qt

# Import your existing modules
try:
    from audio_processor import AudioProcessor
    from speech_to_text import SpeechToText
    from translator import SubtitleTranslator
    from device_utils import print_device_summary
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")


class FileItem:
    """Class to represent a file item in the processing queue"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.status = "pending"  # pending, processing, completed, error
        self.vocal_file = None
        self.srt_file = None
        self.error_message = None


class ExtractionWorker(QThread):
    """Worker thread for subtitle extraction"""
    progress_update = pyqtSignal(str, int)  # stage, percentage
    status_update = pyqtSignal(str)  # status message
    subtitle_segment = pyqtSignal(dict)  # real-time subtitle segment
    file_completed = pyqtSignal(str, bool, dict)  # file_path, success, result_data
    all_completed = pyqtSignal()
    
    def __init__(self, files: List[FileItem], output_dir: str, language: str, 
                 model: str, device: str, skip_vocal: bool):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.language = language
        self.model = model
        self.device = device
        self.skip_vocal = skip_vocal
        self.should_stop = Event()
        
    def stop(self):
        self.should_stop.set()
        
    def run(self):
        try:
            for i, file_item in enumerate(self.files):
                if self.should_stop.is_set():
                    break
                    
                self.status_update.emit(f"Processing file {i+1}/{len(self.files)}: {Path(file_item.file_path).name}")
                
                try:
                    # Process single file
                    result = self._process_single_file(file_item)
                    if result:
                        file_item.status = "completed"
                        file_item.vocal_file = result.get("vocal_file")
                        file_item.srt_file = result.get("srt_file")
                        self.file_completed.emit(file_item.file_path, True, result)
                    else:
                        file_item.status = "error"
                        file_item.error_message = "Processing failed"
                        self.file_completed.emit(file_item.file_path, False, {"error": "Processing failed"})
                        
                except Exception as e:
                    file_item.status = "error"
                    file_item.error_message = str(e)
                    self.file_completed.emit(file_item.file_path, False, {"error": str(e)})
                    
            self.all_completed.emit()
            
        except Exception as e:
            self.status_update.emit(f"Fatal error: {str(e)}")
            
    def _process_single_file(self, file_item: FileItem) -> Optional[Dict[str, Any]]:
        """Process a single file through the pipeline"""
        try:
            input_path = Path(file_item.file_path)
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize processors
            audio_processor = AudioProcessor(
                device=self.device, 
                progress_callback=self._progress_callback
            )
            
            stt = SpeechToText(
                engine="faster_whisper",
                model_size=self.model,
                device=self.device,
                progress_callback=self._progress_callback,
                segment_callback=self._segment_callback,
                log_callback=self._log_callback
            )
            
            # Step 1: Audio preparation/vocal separation
            audio_file: Path
            if self.skip_vocal:
                self.status_update.emit("⏭️ Skipping vocal separation → Using original audio")
                temp_audio_path = output_dir / f"{input_path.stem}_temp.wav"
                
                # Check if input is video
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
                if input_path.suffix.lower() in video_extensions:
                    self.status_update.emit("🎬 Extracting audio from video...")
                    if not audio_processor.extract_audio_from_video(str(input_path), str(temp_audio_path)):
                        raise RuntimeError("Audio extraction failed")
                    audio_file = temp_audio_path
                else:
                    audio_file = input_path
            else:
                self.status_update.emit("🎤 Separating vocals...")
                vocal_file = audio_processor.process_file(str(input_path), str(output_dir))
                if not vocal_file:
                    raise RuntimeError("Audio processing failed")
                audio_file = Path(vocal_file)
            
            # Step 2: Speech recognition → SRT
            self.status_update.emit("📝 Performing speech recognition...")
            original_srt = output_dir / f"{input_path.stem}_original.srt"
            
            # Handle language parameter
            lang_arg = None if (self.language in (None, "", "auto")) else self.language
            
            if not stt.transcribe_to_srt(str(audio_file), str(original_srt), language=lang_arg):
                raise RuntimeError("Speech recognition failed")
                
            self.status_update.emit("🎉 Subtitle extraction completed")
            
            # Cleanup
            stt.close()
            audio_processor.close()
            
            return {
                "vocal_file": str(audio_file) if not self.skip_vocal else None,
                "srt_file": str(original_srt)
            }
            
        except Exception as e:
            self.status_update.emit(f"❌ Error processing {input_path.name}: {str(e)}")
            raise e
            
    def _progress_callback(self, tag: str, value: float):
        """Callback for progress updates"""
        self.progress_update.emit(tag, int(value))
        
    def _segment_callback(self, segment: dict):
        """Callback for real-time subtitle segments"""
        self.subtitle_segment.emit(segment)
        
    def _log_callback(self, message: str):
        """Callback for log messages"""
        self.status_update.emit(message)


class TranslationWorker(QThread):
    """Worker thread for subtitle translation"""
    progress_update = pyqtSignal(str)  # streaming translation content
    status_update = pyqtSignal(str)  # status message
    file_completed = pyqtSignal(str, bool)  # file_path, success
    all_completed = pyqtSignal()
    
    def __init__(self, srt_files: List[str], api_key: str, source_lang: str, 
                 target_lang: str, program_name: str, program_notes: str, output_dir: str):
        super().__init__()
        self.srt_files = srt_files
        self.api_key = api_key
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.program_name = program_name
        self.program_notes = program_notes
        self.output_dir = output_dir
        self.should_stop = Event()
        
    def stop(self):
        self.should_stop.set()
        
    def run(self):
        try:
            for i, srt_file in enumerate(self.srt_files):
                if self.should_stop.is_set():
                    break
                    
                self.status_update.emit(f"Translating file {i+1}/{len(self.srt_files)}: {Path(srt_file).name}")
                
                try:
                    # Translate single file
                    success = self._translate_single_file(srt_file)
                    self.file_completed.emit(srt_file, success)
                    
                except Exception as e:
                    self.status_update.emit(f"❌ Error translating {Path(srt_file).name}: {str(e)}")
                    self.file_completed.emit(srt_file, False)
                    
            self.all_completed.emit()
            
        except Exception as e:
            self.status_update.emit(f"Fatal error: {str(e)}")
            
    def _translate_single_file(self, srt_file: str) -> bool:
        """Translate a single SRT file"""
        try:
            input_path = Path(srt_file)
            output_path = Path(self.output_dir) / f"{input_path.stem}_translated_{self.target_lang.lower()}.srt"
            
            # Initialize translator
            translator = SubtitleTranslator(
                api_key=self.api_key,
                target_language=self.target_lang,
                batch_size=100,
                program_name=self.program_name,
                additional_context=self.program_notes
            )
            
            # Translate with streaming callback
            success = translator.translate_srt_file(
                str(input_path), 
                str(output_path), 
                self.program_name,
                stream_callback=self._stream_callback
            )
            
            translator.close()
            
            if success:
                self.status_update.emit(f"✅ Translation completed: {output_path.name}")
            else:
                self.status_update.emit(f"❌ Translation failed: {input_path.name}")
                
            return success
            
        except Exception as e:
            self.status_update.emit(f"❌ Error: {str(e)}")
            return False
            
    def _stream_callback(self, text: str):
        """Callback for streaming translation updates"""
        self.progress_update.emit(text)


class SubtitleExtractionTab(QWidget):
    """Tab for subtitle extraction functionality"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.selected_files: List[FileItem] = []
        self.completed_srt_files: List[str] = []
        self.extraction_worker = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # File selection buttons
        button_layout = QHBoxLayout()
        self.select_files_btn = QPushButton("Select Video/Audio Files")
        self.select_files_btn.setMinimumHeight(40)
        self.clear_files_btn = QPushButton("Clear All")
        self.clear_files_btn.setMinimumHeight(40)
        
        button_layout.addWidget(self.select_files_btn)
        button_layout.addWidget(self.clear_files_btn)
        file_layout.addLayout(button_layout)
        
        # Selected files list
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(150)
        file_layout.addWidget(QLabel("Selected Files:"))
        file_layout.addWidget(self.files_list)
        
        layout.addWidget(file_group)
        
        # Settings section
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Output directory
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(str(Path("output").absolute()))
        self.output_dir_btn = QPushButton("Browse")
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_dir_btn)
        settings_layout.addRow("Output Directory:", output_layout)
        
        # Language selection
        self.language_combo = QComboBox()
        self._load_languages()
        settings_layout.addRow("Source Language:", self.language_combo)
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["large-v3", "medium", "small", "base"])
        settings_layout.addRow("Whisper Model:", self.model_combo)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        settings_layout.addRow("Device:", self.device_combo)
        
        # Skip vocal separation option
        self.skip_vocal_checkbox = QCheckBox("Skip vocal separation (use original audio)")
        settings_layout.addRow(self.skip_vocal_checkbox)
        
        layout.addWidget(settings_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Progress bars
        self.progress_ffmpeg = QProgressBar()
        self.progress_uvr = QProgressBar()
        self.progress_whisper_load = QProgressBar()
        self.progress_whisper_transcribe = QProgressBar()
        
        for bar in [self.progress_ffmpeg, self.progress_uvr, 
                   self.progress_whisper_load, self.progress_whisper_transcribe]:
            bar.setRange(0, 100)
            bar.setValue(0)
            
        progress_layout.addWidget(QLabel("FFmpeg Audio Extraction:"))
        progress_layout.addWidget(self.progress_ffmpeg)
        progress_layout.addWidget(QLabel("Kim_vocal Model Download/Processing:"))
        progress_layout.addWidget(self.progress_uvr)
        progress_layout.addWidget(QLabel("Whisper Model Loading:"))
        progress_layout.addWidget(self.progress_whisper_load)
        progress_layout.addWidget(QLabel("Whisper Transcription:"))
        progress_layout.addWidget(self.progress_whisper_transcribe)
        
        layout.addWidget(progress_group)
        
        # Real-time subtitle display
        subtitle_group = QGroupBox("Real-time Subtitles")
        subtitle_layout = QVBoxLayout(subtitle_group)
        
        self.subtitle_display = QPlainTextEdit()
        self.subtitle_display.setReadOnly(True)
        self.subtitle_display.setMaximumHeight(150)
        self.subtitle_display.setPlaceholderText("Real-time subtitles will appear here...")
        subtitle_layout.addWidget(self.subtitle_display)
        
        layout.addWidget(subtitle_group)
        
        # Status and control
        control_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        control_layout.addWidget(self.status_label)
        
        self.start_extraction_btn = QPushButton("Start Extraction")
        self.start_extraction_btn.setMinimumHeight(50)
        self.start_extraction_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.start_extraction_btn)
        
        layout.addLayout(control_layout)
        
    def setup_connections(self):
        self.select_files_btn.clicked.connect(self.select_files)
        self.clear_files_btn.clicked.connect(self.clear_files)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        self.start_extraction_btn.clicked.connect(self.start_extraction)
        
    def _load_languages(self):
        """Load language options from languages.json"""
        json_path = Path("languages.json")
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                src_codes = data.get("source_languages", [])
            except Exception:
                src_codes = ["auto", "en", "ko", "ja", "zh", "es", "fr", "de"]
        else:
            src_codes = ["auto", "en", "ko", "ja", "zh", "es", "fr", "de"]
            
        # Language code to name mapping
        name_map = {
            "auto": "Auto Detect", "af": "Afrikaans", "am": "Amharic", "ar": "Arabic",
            "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian",
            "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "br": "Breton",
            "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
            "da": "Danish", "de": "German", "el": "Greek", "en": "English",
            "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian",
            "fi": "Finnish", "fo": "Faroese", "fr": "French", "gl": "Galician",
            "gu": "Gujarati", "ha": "Hausa", "he": "Hebrew", "hi": "Hindi",
            "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
            "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese",
            "ka": "Georgian", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
            "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala",
            "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy",
            "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
            "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
            "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "oc": "Occitan",
            "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
            "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit", "sd": "Sindhi",
            "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona",
            "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese",
            "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
            "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog",
            "tr": "Turkish", "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu",
            "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba",
            "zh": "Chinese"
        }
        
        display_names = [name_map.get(code, code.upper()) for code in src_codes]
        self.language_combo.addItems(display_names)
        self.language_combo.setProperty("_codes", src_codes)
        
    def select_files(self):
        """Select multiple video/audio files"""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter(
            "Media Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v "
            "*.mp3 *.wav *.flac *.aac *.ogg *.m4a);;All Files (*)"
        )
        
        if file_dialog.exec():
            selected_paths = file_dialog.selectedFiles()
            for path in selected_paths:
                # Check if file already exists in list
                if not any(item.file_path == path for item in self.selected_files):
                    file_item = FileItem(path)
                    self.selected_files.append(file_item)
                    
            self.update_files_display()
            
    def clear_files(self):
        """Clear all selected files"""
        self.selected_files.clear()
        self.update_files_display()
        
    def update_files_display(self):
        """Update the files list display"""
        self.files_list.clear()
        for file_item in self.selected_files:
            item_text = f"{Path(file_item.file_path).name} - {file_item.status}"
            list_item = QListWidgetItem(item_text)
            
            # Color code by status
            if file_item.status == "completed":
                list_item.setBackground(QtGui.QColor(200, 255, 200))
            elif file_item.status == "error":
                list_item.setBackground(QtGui.QColor(255, 200, 200))
            elif file_item.status == "processing":
                list_item.setBackground(QtGui.QColor(255, 255, 200))
                
            self.files_list.addItem(list_item)
            
    def select_output_dir(self):
        """Select output directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)
            
    def start_extraction(self):
        """Start the extraction process"""
        if not self.selected_files:
            QMessageBox.warning(self, "Warning", "Please select at least one file.")
            return
            
        # Get settings
        output_dir = self.output_dir_edit.text().strip() or str(Path("output").absolute())
        
        # Get language code
        src_codes = self.language_combo.property("_codes") or []
        src_idx = self.language_combo.currentIndex()
        language = src_codes[src_idx] if 0 <= src_idx < len(src_codes) else "auto"
        
        model = self.model_combo.currentText()
        device = self.device_combo.currentText()
        skip_vocal = self.skip_vocal_checkbox.isChecked()
        
        # Reset progress bars and status
        for bar in [self.progress_ffmpeg, self.progress_uvr, 
                   self.progress_whisper_load, self.progress_whisper_transcribe]:
            bar.setValue(0)
            
        self.subtitle_display.clear()
        self.status_label.setText("Starting extraction...")
        self.start_extraction_btn.setEnabled(False)
        
        # Reset file statuses
        for file_item in self.selected_files:
            file_item.status = "pending"
        self.update_files_display()
        
        # Start worker thread
        self.extraction_worker = ExtractionWorker(
            self.selected_files, output_dir, language, model, device, skip_vocal
        )
        
        self.extraction_worker.progress_update.connect(self.update_progress)
        self.extraction_worker.status_update.connect(self.update_status)
        self.extraction_worker.subtitle_segment.connect(self.update_subtitle_display)
        self.extraction_worker.file_completed.connect(self.on_file_completed)
        self.extraction_worker.all_completed.connect(self.on_all_completed)
        
        self.extraction_worker.start()
        
    def update_progress(self, stage: str, percentage: int):
        """Update progress bars based on stage"""
        if "ffmpeg" in stage.lower():
            self.progress_ffmpeg.setValue(percentage)
        elif "uvr" in stage.lower() or "kim_vocal" in stage.lower():
            self.progress_uvr.setValue(percentage)
        elif "whisper_load" in stage.lower() or "whisper_download" in stage.lower():
            self.progress_whisper_load.setValue(percentage)
        elif "whisper_transcribe" in stage.lower():
            self.progress_whisper_transcribe.setValue(percentage)
            
    def update_status(self, message: str):
        """Update status label"""
        self.status_label.setText(message)
        
    def update_subtitle_display(self, segment: dict):
        """Update real-time subtitle display"""
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('text', '')
        
        subtitle_line = f"[{start_time:.2f} → {end_time:.2f}] {text}"
        self.subtitle_display.appendPlainText(subtitle_line)
        
    def on_file_completed(self, file_path: str, success: bool, result_data: dict):
        """Handle completion of a single file"""
        # Update file status
        for file_item in self.selected_files:
            if file_item.file_path == file_path:
                if success:
                    file_item.status = "completed"
                    if result_data.get("srt_file"):
                        self.completed_srt_files.append(result_data["srt_file"])
                else:
                    file_item.status = "error"
                    file_item.error_message = result_data.get("error", "Unknown error")
                break
                
        self.update_files_display()
        
    def on_all_completed(self):
        """Handle completion of all files"""
        self.status_label.setText("All extractions completed!")
        self.start_extraction_btn.setEnabled(True)
        
        # Show completion alert
        completed_count = sum(1 for f in self.selected_files if f.status == "completed")
        error_count = sum(1 for f in self.selected_files if f.status == "error")
        
        message = f"Extraction completed!\n\nSuccessful: {completed_count}\nErrors: {error_count}"
        QMessageBox.information(self, "Extraction Complete", message)
        
        # Notify parent window about completed SRT files
        if hasattr(self.parent_window, 'translation_tab'):
            self.parent_window.translation_tab.update_available_srt_files(self.completed_srt_files)


class TranslationTab(QWidget):
    """Tab for subtitle translation functionality"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.available_srt_files: List[str] = []
        self.translation_worker = None
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # API Configuration section
        api_group = QGroupBox("API Configuration")
        api_layout = QFormLayout(api_group)
        
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Enter your Gemini API key or set GEMINI_API_KEY environment variable")
        api_layout.addRow("Gemini API Key:", self.api_key_edit)
        
        layout.addWidget(api_group)
        
        # Program Information section
        program_group = QGroupBox("Program Information")
        program_layout = QFormLayout(program_group)
        
        self.program_name_edit = QLineEdit()
        self.program_name_edit.setPlaceholderText("e.g., RADWIMPS - 前前前世")
        program_layout.addRow("Program/Song Title:", self.program_name_edit)
        
        self.program_notes_edit = QTextEdit()
        self.program_notes_edit.setMaximumHeight(100)
        self.program_notes_edit.setPlaceholderText("Additional context, notes, or description...")
        program_layout.addRow("Additional Notes:", self.program_notes_edit)
        
        layout.addWidget(program_group)
        
        # Language Selection section
        lang_group = QGroupBox("Language Selection")
        lang_layout = QFormLayout(lang_group)
        
        self.source_lang_combo = QComboBox()
        self.target_lang_combo = QComboBox()
        
        self._load_target_languages()
        
        lang_layout.addRow("Source Language:", self.source_lang_combo)
        lang_layout.addRow("Target Language:", self.target_lang_combo)
        
        layout.addWidget(lang_group)
        
        # SRT Files section
        files_group = QGroupBox("SRT Files")
        files_layout = QVBoxLayout(files_group)
        
        # Manual SRT file selection
        manual_layout = QHBoxLayout()
        self.manual_srt_edit = QLineEdit()
        self.manual_srt_edit.setPlaceholderText("Select SRT file manually...")
        self.browse_srt_btn = QPushButton("Browse")
        manual_layout.addWidget(self.manual_srt_edit)
        manual_layout.addWidget(self.browse_srt_btn)
        files_layout.addLayout(manual_layout)
        
        # Available SRT files from extraction
        files_layout.addWidget(QLabel("Available SRT files from extraction:"))
        self.srt_files_list = QListWidget()
        self.srt_files_list.setMaximumHeight(150)
        self.srt_files_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        files_layout.addWidget(self.srt_files_list)
        
        layout.addWidget(files_group)
        
        # Translation Progress section
        progress_group = QGroupBox("Translation Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.translation_status = QPlainTextEdit()
        self.translation_status.setReadOnly(True)
        self.translation_status.setPlaceholderText("Translation progress will be shown here...")
        progress_layout.addWidget(self.translation_status)
        
        layout.addWidget(progress_group)
        
        # Control section
        control_layout = QVBoxLayout()
        
        self.start_translation_btn = QPushButton("Start Translation")
        self.start_translation_btn.setMinimumHeight(50)
        self.start_translation_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        control_layout.addWidget(self.start_translation_btn)
        
        layout.addLayout(control_layout)
        
    def setup_connections(self):
        self.browse_srt_btn.clicked.connect(self.browse_srt_file)
        self.start_translation_btn.clicked.connect(self.start_translation)
        
    def _load_target_languages(self):
        """Load target language options"""
        json_path = Path("languages.json")
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                target_langs = data.get("target_languages", [])
                source_codes = data.get("source_languages", [])
            except Exception:
                target_langs = ["Korean", "English", "Japanese", "Chinese", "Spanish", "French", "German"]
                source_codes = ["auto", "en", "ko", "ja", "zh", "es", "fr", "de"]
        else:
            target_langs = ["Korean", "English", "Japanese", "Chinese", "Spanish", "French", "German"]
            source_codes = ["auto", "en", "ko", "ja", "zh", "es", "fr", "de"]
            
        # Load source languages (same as extraction tab)
        name_map = {
            "auto": "Auto Detect", "en": "English", "ko": "Korean", "ja": "Japanese",
            "zh": "Chinese", "es": "Spanish", "fr": "French", "de": "German"
        }
        
        source_display_names = [name_map.get(code, code.upper()) for code in source_codes]
        self.source_lang_combo.addItems(source_display_names)
        self.source_lang_combo.setProperty("_codes", source_codes)
        
        self.target_lang_combo.addItems(target_langs)
        
    def browse_srt_file(self):
        """Browse for SRT file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select SRT File", "", "SRT Files (*.srt);;All Files (*)"
        )
        if file_path:
            self.manual_srt_edit.setText(file_path)
            
    def update_available_srt_files(self, srt_files: List[str]):
        """Update the list of available SRT files from extraction"""
        self.available_srt_files = srt_files
        self.srt_files_list.clear()
        
        for srt_file in srt_files:
            item = QListWidgetItem(Path(srt_file).name)
            item.setData(Qt.ItemDataRole.UserRole, srt_file)
            self.srt_files_list.addItem(item)
            
    def get_selected_srt_files(self) -> List[str]:
        """Get list of selected SRT files for translation"""
        selected_files = []
        
        # Add manually selected file
        manual_file = self.manual_srt_edit.text().strip()
        if manual_file and Path(manual_file).exists():
            selected_files.append(manual_file)
            
        # Add selected files from list
        for item in self.srt_files_list.selectedItems():
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path and file_path not in selected_files:
                selected_files.append(file_path)
                
        return selected_files
        
    def start_translation(self):
        """Start the translation process"""
        # Validate inputs
        api_key = self.api_key_edit.text().strip() or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            QMessageBox.warning(self, "Warning", "Please provide a Gemini API key.")
            return
            
        selected_srt_files = self.get_selected_srt_files()
        if not selected_srt_files:
            QMessageBox.warning(self, "Warning", "Please select at least one SRT file.")
            return
            
        program_name = self.program_name_edit.text().strip()
        program_notes = self.program_notes_edit.toPlainText().strip()
        
        # Get language settings
        src_codes = self.source_lang_combo.property("_codes") or []
        src_idx = self.source_lang_combo.currentIndex()
        source_lang = src_codes[src_idx] if 0 <= src_idx < len(src_codes) else "auto"
        
        target_lang = self.target_lang_combo.currentText()
        
        # Get output directory (same as extraction tab)
        if hasattr(self.parent_window, 'extraction_tab'):
            output_dir = self.parent_window.extraction_tab.output_dir_edit.text().strip()
        else:
            output_dir = str(Path("output").absolute())
            
        # Clear translation status
        self.translation_status.clear()
        self.start_translation_btn.setEnabled(False)
        
        # Start translation worker
        self.translation_worker = TranslationWorker(
            selected_srt_files, api_key, source_lang, target_lang,
            program_name, program_notes, output_dir
        )
        
        self.translation_worker.progress_update.connect(self.update_translation_progress)
        self.translation_worker.status_update.connect(self.update_translation_status)
        self.translation_worker.file_completed.connect(self.on_translation_file_completed)
        self.translation_worker.all_completed.connect(self.on_translation_all_completed)
        
        self.translation_worker.start()
        
    def update_translation_progress(self, text: str):
        """Update translation progress with streaming content"""
        self.translation_status.appendPlainText(text)
        
    def update_translation_status(self, message: str):
        """Update translation status"""
        self.translation_status.appendPlainText(f"[STATUS] {message}")
        
    def on_translation_file_completed(self, file_path: str, success: bool):
        """Handle completion of a single file translation"""
        file_name = Path(file_path).name
        if success:
            self.translation_status.appendPlainText(f"✅ Completed: {file_name}")
        else:
            self.translation_status.appendPlainText(f"❌ Failed: {file_name}")
            
    def on_translation_all_completed(self):
        """Handle completion of all translations"""
        self.translation_status.appendPlainText("\n🎉 All translations completed!")
        self.start_translation_btn.setEnabled(True)
        
        # Show completion alert
        QMessageBox.information(self, "Translation Complete", "All subtitle translations have been completed!")


class MainWindow(QMainWindow):
    """Main application window with tabbed interface"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Gemini Subtitle - Extraction & Translation")
        self.setMinimumSize(1000, 800)
        
        # Set application icon if available
        try:
            self.setWindowIcon(QIcon("icon.png"))
        except:
            pass
            
        # Create central widget with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.extraction_tab = SubtitleExtractionTab(self)
        self.translation_tab = TranslationTab(self)
        
        # Add tabs with English names as requested
        self.tabs.addTab(self.extraction_tab, "Subtitle Extraction")
        self.tabs.addTab(self.translation_tab, "Translation")
        
        # Set modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: black;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                font-size: 12px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196F3;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 10px 0px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
    def closeEvent(self, event):
        """Handle application close event"""
        # Stop any running workers
        if hasattr(self.extraction_tab, 'extraction_worker') and self.extraction_tab.extraction_worker:
            self.extraction_tab.extraction_worker.stop()
            self.extraction_tab.extraction_worker.wait(3000)
            
        if hasattr(self.translation_tab, 'translation_worker') and self.translation_tab.translation_worker:
            self.translation_tab.translation_worker.stop()
            self.translation_tab.translation_worker.wait(3000)
            
        event.accept()


def main():
    """Main application entry point"""
    # Enable high DPI support (PyQt6 handles this automatically)
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Gemini Subtitle")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Gemini Subtitle")
    
    # Set application style for cross-platform compatibility
    if platform.system() == "Windows":
        app.setStyle("windowsvista")
    elif platform.system() == "Darwin":  # macOS
        app.setStyle("macos")
    else:  # Linux
        app.setStyle("fusion")
        
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Center window on screen
    screen = app.primaryScreen().geometry()
    size = window.geometry()
    window.move(
        (screen.width() - size.width()) // 2,
        (screen.height() - size.height()) // 2
    )
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
