
import pandas as pd
import PyPDF2
from docx import Document as DocxDocument
from typing import List
from langchain.schema import Document
from pathlib import Path
from utils.logger import logger


class DocumentParser:
    """Parse multiple document formats into LangChain Documents"""
    
    @staticmethod
    def parse_excel(file_path: str) -> List[Document]:
        """
        Parse Excel file
        
        Args:
            file_path: Path to Excel file
        
        Returns:
            List of LangChain Document objects
        """
        documents = []
        excel_file = pd.ExcelFile(file_path)
        filename = Path(file_path).name
        
        logger.info(f"📊 Parsing Excel: {filename}")
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Sheet summary
            summary_content = f"Excel File: {filename}\n"
            summary_content += f"Sheet: {sheet_name}\n"
            summary_content += f"Columns: {', '.join(df.columns.tolist())}\n"
            summary_content += f"Total Rows: {len(df)}\n\n"
            
            # Add sample data
            if len(df) > 0:
                summary_content += "Sample Data:\n"
                summary_content += df.head(3).to_string()
            
            metadata = {
                "source": filename,
                "type": "excel",
                "sheet": sheet_name,
                "row_count": len(df),
                "columns": list(df.columns)
            }
            
            documents.append(Document(
                page_content=summary_content,
                metadata=metadata
            ))
            
            # Add individual rows as documents
            for idx, row in df.iterrows():
                content = f"From {filename}, Sheet '{sheet_name}', Row {idx}:\n"
                content += ", ".join([f"{col}: {val}" for col, val in row.items()])
                
                row_metadata = {
                    "source": filename,
                    "type": "excel_row",
                    "sheet": sheet_name,
                    "row": int(idx)
                }
                
                documents.append(Document(
                    page_content=content,
                    metadata=row_metadata
                ))
        
        logger.info(f"✅ Parsed {len(documents)} chunks from {filename}")
        return documents
    
    @staticmethod
    def parse_txt(file_path: str) -> List[Document]:
        """Parse text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return [Document(
            page_content=text,
            metadata={"source": Path(file_path).name, "type": "txt"}
        )]
    
    @staticmethod
    def parse_csv(file_path: str) -> List[Document]:
        
        documents = []
        df = pd.read_csv(file_path)
        filename = Path(file_path).name
        
        logger.info(f"📄 Parsing CSV: {filename}")
        
        # CSV summary
        summary_content = f"CSV File: {filename}\n"
        summary_content += f"Columns: {', '.join(df.columns.tolist())}\n"
        summary_content += f"Total Rows: {len(df)}\n\n"
        summary_content += "Sample Data:\n"
        summary_content += df.head(5).to_string()
        
        metadata = {
            "source": filename,
            "type": "csv_summary",
            "row_count": len(df),
            "columns": list(df.columns)
        }
        
        documents.append(Document(
            page_content=summary_content,
            metadata=metadata
        ))
        
        # Individual rows
        for idx, row in df.iterrows():
            content = f"From {filename}, Row {idx}:\n"
            content += ", ".join([f"{col}: {val}" for col, val in row.items()])
            
            row_metadata = {
                "source": filename,
                "type": "csv_row",
                "row": int(idx)
            }
            
            documents.append(Document(
                page_content=content,
                metadata=row_metadata
            ))
        
        logger.info(f"✅ Parsed {len(documents)} chunks from {filename}")
        return documents
    
    @staticmethod
    def parse_pdf(file_path: str) -> List[Document]:
        
        documents = []
        filename = Path(file_path).name
        
        logger.info(f"📕 Parsing PDF: {filename}")
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # PDF summary
            summary_text = f"PDF Document: {filename}\nTotal Pages: {total_pages}"
            documents.append(Document(
                page_content=summary_text,
                metadata={
                    "source": filename,
                    "type": "pdf_summary",
                    "total_pages": total_pages
                }
            ))
            
            # Individual pages
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():
                    content = f"From {filename}, Page {page_num + 1}:\n{text.strip()}"
                    
                    metadata = {
                        "source": filename,
                        "type": "pdf",
                        "page": page_num + 1,
                        "total_pages": total_pages
                    }
                    
                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
        
        logger.info(f"✅ Parsed {len(documents)} chunks from {filename}")
        return documents
    
    @staticmethod
    def parse_docx(file_path: str) -> List[Document]:
        
        doc = DocxDocument(file_path)
        documents = []
        filename = Path(file_path).name
        
        logger.info(f"📝 Parsing Word Doc: {filename}")
        
        # Document summary
        paragraph_count = len([p for p in doc.paragraphs if p.text.strip()])
        table_count = len(doc.tables)
        
        summary_text = f"Word Document: {filename}\n"
        summary_text += f"Paragraphs: {paragraph_count}\n"
        summary_text += f"Tables: {table_count}\n"
        
        documents.append(Document(
            page_content=summary_text,
            metadata={
                "source": filename,
                "type": "docx_summary",
                "paragraph_count": paragraph_count,
                "table_count": table_count
            }
        ))
        
        # Paragraphs
        for idx, para in enumerate(doc.paragraphs):
            if para.text.strip():
                content = f"From {filename}, Paragraph {idx}:\n{para.text.strip()}"
                
                metadata = {
                    "source": filename,
                    "type": "docx_paragraph",
                    "paragraph": idx
                }
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
        
        # Tables
        for table_idx, table in enumerate(doc.tables):
            table_text = f"From {filename}, Table {table_idx}:\n"
            
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                table_text += " | ".join(cells) + "\n"
            
            metadata = {
                "source": filename,
                "type": "docx_table",
                "table": table_idx
            }
            
            documents.append(Document(
                page_content=table_text,
                metadata=metadata
            ))
        
        logger.info(f"✅ Parsed {len(documents)} chunks from {filename}")
        return documents