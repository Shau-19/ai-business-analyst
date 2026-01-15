# tools/language_detector.py
"""
Language detection tool
Detects the language of input text
"""
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
        """
        Detect language of text
        
        Args:
            text: Input text
        
        Returns:
            Language code (e.g., 'en', 'es', 'fr') or None if detection fails
        """
        try:
            lang_code = detect(text)
            logger.info(f"🌍 Detected language: {self.get_language_name(lang_code)} ({lang_code})")
            return lang_code
        except LangDetectException:
            logger.warning("⚠️  Could not detect language, defaulting to English")
            return 'en'
    
    def get_language_name(self, lang_code: str) -> str:
        """
        Get full language name from code
        
        Args:
            lang_code: Language code (e.g., 'en')
        
        Returns:
            Full language name (e.g., 'English')
        """
        return self.LANGUAGE_NAMES.get(lang_code, lang_code.upper())
    
    def is_supported(self, lang_code: str) -> bool:
        """
        Check if language is supported
        
        Args:
            lang_code: Language code
        
        Returns:
            True if supported, False otherwise
        """
        return lang_code in self.LANGUAGE_NAMES