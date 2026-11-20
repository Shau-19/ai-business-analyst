import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({ breaks: true, gfm: true });

// Answers can quote text from user-uploaded documents, and marked passes raw
// HTML through untouched — so `<img src=x onerror=...>` inside a PDF/DOCX
// could execute in the viewer's browser. Always sanitize before rendering
// with dangerouslySetInnerHTML.
export function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || ''), {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['style', 'form', 'input', 'button', 'iframe'],
  });
}

// Links in answers open in a new tab without giving it access to this page.
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A' && node.getAttribute('href')) {
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  }
});
