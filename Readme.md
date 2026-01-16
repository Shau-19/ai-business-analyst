# 🤖 AI Business Analyst - Multilingual Hybrid NLP System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)

**Intelligent query routing system combining SQL precision with RAG semantic understanding**

**91.4% Overall Performance** • **9 Languages** • **Hybrid CSV Processing**

[Quick Start](#-quick-start) • [Features](#-key-features) • [Demo](#-usage-examples) • [Results](#-evaluation-results)

</div>

---

## 📊 Evaluation Results

Comprehensive evaluation on 23 benchmark queries demonstrates production-ready performance:

<div align="center">

<img src="evaluation/figures/dashboard.png" alt="System Dashboard" width="100%"/>

| Metric | Score | Status |
|--------|-------|--------|
| **Overall Grade** | **91.4% (A+)** | 🏆 Excellent |
| Routing Accuracy | 83.3% | ✅ Great |
| Language Detection | 90.9% (9 languages) | ✅ Excellent |
| Query Success | 100% | ✅ Perfect |
| Avg Response Time | 1.54s | ⚡ Fast |

</div>

<details>
<summary><b>📈 View Detailed Performance Charts</b></summary>

<table>
<tr>
<td width="50%"><img src="evaluation/figures/overall_metrics.png" alt="Metrics"/></td>
<td width="50%"><img src="evaluation/figures/confusion_matrix.png" alt="Confusion Matrix"/></td>
</tr>
<tr>
<td width="50%"><img src="evaluation/figures/language_coverage.png" alt="Languages"/></td>
<td width="50%"><img src="evaluation/figures/response_times.png" alt="Response Times"/></td>
</tr>
</table>

**Key Findings:**
- ✅ Perfect SQL classification (5/5 queries)
- ✅ Perfect document routing (5/5 queries)
- ✅ Multilingual: English, Spanish, French, German, Hindi, Japanese, Arabic, Portuguese, Korean
- ✅ Consistent sub-2s response times

</details>

---

## 🎯 Key Features

### 🔀 **Intelligent Hybrid Processing**

Unlike traditional systems that force SQL-only or RAG-only approaches, our system intelligently uses **both**:

```
                    Document Upload
                          │
              ┌───────────┴───────────────┐
              │                           │
         PDF/Word/TXT              CSV/Excel
              │                           │
              ▼                           ▼
      ┌──────────────┐         ┌────────────────────┐
      │  RAG Only    │         │   Hybrid Mode      │
      │  (Semantic)  │         │  (SQL + RAG)       │
      └──────────────┘         └─────────┬──────────┘
              │                           │
              │                  ┌────────┴────────┐
              │                  │                 │
              │            ┌─────▼──────┐   ┌──────▼─────┐
              │            │ SQL Import │   │ RAG Parse  │
              │            │(Precision) │   │ (Insight)  │
              │            └─────┬──────┘   └──────┬─────┘
              │                  └────────┬─────────┘
              │                           │
              └───────────────┬───────────┘
                              │
                      ┌───────▼────────┐
                      │  Smart Router  │
                      │ (Query Type)   │
                      └───────┬────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
         │   SQL   │    │   RAG   │    │ HYBRID  │
         │"Calculate"   │"Explain"│    │  "Why?" │
         └─────────┘    └─────────┘    └─────────┘
```

**Query Examples:**

| Query Type | Example | Route | Why |
|------------|---------|-------|-----|
| **Analytical** | "What percentage of items are complete?" | SQL | Needs exact calculations |
| **Semantic** | "Summarize the trends in this data" | RAG | Needs understanding |
| **Hybrid** ⭐ | "Why did sales drop 20% in March?" | SQL+RAG | Needs numbers AND context |

**The Innovation:** Questions like "Why did X happen?" require **both** precise calculations (SQL) **and** contextual analysis (RAG). Traditional systems can't do this.

### 🌍 **Multilingual Support (90.9% Accuracy)**

Automatically detects and responds in 9 languages:
- 🇬🇧 English • 🇪🇸 Spanish • 🇫🇷 French • 🇩🇪 German
- 🇮🇳 Hindi • 🇯🇵 Japanese • 🇸🇦 Arabic • 🇵🇹 Portuguese • 🇰🇷 Korean

```python
# Ask in any language, get answer in the same language
"¿Cuántos empleados hay?" → "Hay 10 empleados en la base de datos."
```

### 🗄️ **Modular Database Design**

Switch databases with **one config change** - no code modifications needed:

| Database | Use Case | Switch With |
|----------|----------|-------------|
| **SQLite** (default) | Development, demos | `DB_TYPE=sqlite` |
| **PostgreSQL** | Production, scalable | `DB_TYPE=postgresql` |
| **MySQL** | Enterprise | `DB_TYPE=mysql` |

Same `DatabaseManager` interface works across all databases.

### 📄 **Session-Aware Document Management**

- **Isolated storage** - Each chat has its own document space
- **Multi-format** - PDF, Word, Excel, CSV, TXT
- **Persistent** - Documents saved per conversation
- **Smart routing** - Automatically selects best approach

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Shau-19/ai-business-analyst.git
cd ai-business-analyst
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (add your GROQ API key)
cp .env.example .env
# Edit .env: GROQ_API_KEY=your_key_here

# 4. Initialize database
python database/sample_data.py

# 5. Run application
python main.py
```

Visit: **http://localhost:8000/ui**

**Requirements:** Python 3.11+, GROQ API key ([get free](https://console.groq.com/keys))

---

## 💡 Usage Examples

### SQL Database Queries
```
"How many employees work in Engineering?"
"What is the average salary by department?"
"Show me total sales for Q4"
```

### Document Queries
```
"What were the action items from the meeting?"
"Summarize the strategic plan"
"What are the key findings in the report?"
```

### Hybrid CSV Queries (Innovation!)
```
"What percentage of tasks are completed?"        → SQL (exact calculation)
"Summarize the trends in this sales data"       → RAG (semantic analysis)
"Why did performance drop 15% in March?"         → HYBRID (both!)
```

### Multilingual
```
"¿Cuántos empleados hay?"              (Spanish)
"Combien d'employés travaillent ici?" (French)
"यहाँ कितने कर्मचारी हैं?"              (Hindi)
```

---

## 🏗️ Architecture

### System Components

```
User Query → Orchestrator → Smart Router
                  ↓
    ┌─────────────┼─────────────┐
    │             │             │
SQL Agent     RAG Agent    Hybrid Mode
    │             │             │
Database    Vector Store   Both!
```

**Key Agents:**
- **Orchestrator** - Routes queries intelligently (83.3% accuracy)
- **SQL Agent** - Generates and executes SQL with multilingual support
- **RAG Agent** - Semantic search with session isolation
- **Hybrid Mode** - Combines SQL precision + RAG insights for complex queries

### Technology Stack

**Core:** Python 3.11 • LangChain • FastAPI  
**AI/ML:** Groq (Llama 3.3 70B) • HuggingFace Embeddings • FAISS  
**Database:** SQLite • PostgreSQL • MySQL (modular, configurable)  
**Protocols:** A2A (Agent-to-Agent) • MCP (Model Context Protocol)

---

## 🧪 Testing

```bash
# Run full evaluation (91.4% grade)
python tests/test_full_system.py

# Generate performance visualizations
python tests/test_visual.py

# Test individual components
python database/db_manager.py    # Database connection
python tools/csv_query_router.py # CSV routing logic
python config.py                 # Configuration
```

Results saved to: `evaluation/figures/*.png`

---

## 📂 Project Structure

```
agents/          - SQL, RAG, and Orchestrator agents
database/        - Modular DB manager + sample data
parsers/         - Hybrid CSV processor + document parsers
tools/           - Language detection, CSV routing, SQL executor
evaluation/      - Test results, visualizations, benchmarks
tests/           - Evaluation framework
mcp/             - Model Context Protocol server
chat/            - Session management with Redis
```

<details>
<summary>View complete structure</summary>

```
ai-business-analyst/
├── agents/
│   ├── orchestrator.py          # Query routing
│   ├── sql_analyst.py           # SQL generation
│   └── session_rag.py           # Document RAG
├── database/
│   ├── db_manager.py            # Modular DB (SQLite/PostgreSQL/MySQL)
│   └── sample_data.py           # Sample data
├── parsers/
│   ├── document_parser.py       # PDF/Word/Excel/CSV parser
│   └── hybrid_csv_processor.py  # Dual SQL+RAG processor
├── tools/
│   ├── language_detector.py     # 9 languages
│   ├── csv_query_router.py      # Intelligent routing
│   └── sql_executor.py          # SQL execution
├── protocols/
│   └── a2a.py                   # Agent-to-Agent
├── mcp/
│   ├── mcp_server.py            # MCP server
│   └── mcp_tools.py             # Tool handlers
├── chat/
│   └── chat_history.py          # Session management
├── evaluation/
│   ├── results.json             # Latest results
│   └── figures/                 # Visualizations
└── tests/
    ├── test_full_system.py      # Full evaluation
    ├── test_visual.py           # Chart generator
    └── test_system.py           # Framework
```

</details>

---

## 🗺️ What's Next

**Current (v1.0):**
- ✅ 9 languages with 90.9% detection
- ✅ Hybrid CSV processing (SQL + RAG)
- ✅ Modular database (SQLite/PostgreSQL/MySQL)
- ✅ 91.4% overall performance

**Planned (v2.0):**
- Enterprise authentication & RBAC
- Central document repository for organizations
- Fine-tuned routing model (83% → 95%)
- Docker containerization
- Advanced analytics dashboard

See [ROADMAP.md](docs/ROADMAP.md) for details.

---

## 📈 Performance

**Average Response Time:** 1.54s (SQLite, local)  
**Query Success Rate:** 100% (23/23 test queries)  
**Evaluation Grade:** 91.4% (A+ Excellent)

Tested on: Apple M1 Pro, 16GB RAM

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push and open Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

**Built with:** [Anthropic MCP](https://modelcontextprotocol.io) • [LangChain](https://langchain.com) • [Groq](https://groq.com) • [HuggingFace](https://huggingface.co)

---

## 📚 Citation

```bibtex
@software{ai_business_analyst_2026,
  author = {Shaurya Jain},
  title = {AI Business Analyst: Multilingual Hybrid NLP System},
  year = {2026},
  url = {https://github.com/Shau-19/ai-business-analyst},
  note = {Hybrid CSV processing combining SQL precision with RAG semantic understanding}
}
```

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

[Report Bug](https://github.com/Shau-19/ai-business-analyst/issues) • [Request Feature](https://github.com/Shau-19/ai-business-analyst/issues)

Made with ❤️ for NLP Research

</div>
