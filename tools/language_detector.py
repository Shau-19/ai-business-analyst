'''Detects language of input text using langdetect library, with support for multiple languages 
and graceful fallback to English if detection fails.'''

from langdetect import detect, LangDetectException
from typing import Optional
from utils.logger import logger


class LanguageDetector:
    """Detect language of text"""
    
    # Language code to full name mapping
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh-cn': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'ur': 'Urdu',
        'pa': 'Punjabi',
        'nl': 'Dutch',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'pl': 'Polish'
    }
    
    def detect_language(self, text: str) -> Optional[str]:
        '''Detect lang (Input_tex -> lang_code(en,fr,hi, etc)'''
        try:
            lang_code = detect(text)
            logger.info(f"🌍 Detected language: {self.get_language_name(lang_code)} ({lang_code})")
            return lang_code
        except LangDetectException:
            logger.warning("⚠️  Could not detect language, defaulting to English")
            return 'en'
    
    def get_language_name(self, lang_code: str) -> str:
        '''Lang_code -> Full language name'''
        return self.LANGUAGE_NAMES.get(lang_code, lang_code.upper())
    
    def is_supported(self, lang_code: str) -> bool:
        '''Check if lang_code is valid or not'''
        return lang_code in self.LANGUAGE_NAMES