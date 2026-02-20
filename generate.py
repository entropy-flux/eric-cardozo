from pathlib import Path
import re
import subprocess
import os

ROOT = Path(__file__).parent

INDEX_MD = ROOT / "index.md"
ABSTRACT_MD = ROOT / "abstract.md"
APPENDIX_MD = ROOT / "appendix.md"
OUT = ROOT / "article.tex"  

PREAMBLE = r"""
\documentclass[12pt]{article}
\usepackage[spanish,provide=*]{babel}
\usepackage[utf8]{inputenc} 
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb} 
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{cite}
\usepackage[font=small,labelfont=bf]{caption}

\usepackage{graphicx}
\usepackage[
  font=small,
  labelfont=bf, 
  margin=0.5cm
]{caption}

\setlength{\textfloatsep}{1.2em}
\setlength{\floatsep}{1em}
\setlength{\intextsep}{1em}

\geometry{a4paper, top=3.5cm, bottom=3.5cm, left=3.5cm, right=3.5cm} 
\setlength{\parindent}{0pt}
\setlength{\parskip}{1em}
\title{Supresión de sinapsis}
\author{Eric Cardozo}
\date{\today}
\begin{document}
\maketitle
"""

POSTAMBLE = r"""
\newpage
\bibliographystyle{unsrt}
\bibliography{references}
\end{document}
"""

# ----------------- Utilities -----------------
def sanitize_unicode(text: str) -> str:
    return text.replace("\u200B", "")

def strip_title(text: str) -> str:
    return re.sub(r"^\s*#\s+.*\n+", "", text)

def strip_citation_section(text: str) -> str:
    return re.sub(r"\n##\s+Citation[\s\S]*$", "", text)

# ----------------- Markdown → LaTeX -----------------
def convert_math(text: str):
    equations = []
    def repl(m):
        eq = m.group(1).strip()
        token = f"@@EQ{len(equations)}@@"
        equations.append(eq)
        return token
    text = re.sub(r"\$\$(.*?)\$\$", repl, text, flags=re.S)
    return text, equations

def restore_math(text: str, equations):
    for i, eq in enumerate(equations):
        text = text.replace(f"@@EQ{i}@@", "\\begin{equation}\n" + eq + "\n\\end{equation}")
    return text

def convert_sections(text: str) -> str:
    text = re.sub(r"^###\s+(.*)$", r"\\subsection*{\1}", text, flags=re.M)
    text = re.sub(r"^##\s+(.*)$", r"\\section*{\1}", text, flags=re.M)
    return text

def convert_lists(text: str) -> str:
    lines = text.splitlines()
    out, in_list = [], False
    for line in lines:
        if re.match(r"^\s*-\s+", line) and not line.startswith("@@EQ"):
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(r"  \item " + line.lstrip("- ").strip())
        else:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(line)
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)

def convert_inline_formatting(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.*?)\*", r"\\emph{\1}", text)
    text = re.sub(r"`(.*?)`", r"\\texttt{\1}", text)
    return text

def convert_citations(text: str) -> str:
    return re.sub(r"\[@([^\]]+)\]", lambda m: m.group(0) if m.group(1).startswith("fig:") else f"\\cite{{{m.group(1)}}}", text)

def convert_fig_refs(text: str) -> str:
    return re.sub(r"\[@fig:([^\]]+)\]", r"\\ref{fig:\1}", text)

def convert_images(text: str) -> str:
    def repl(m):
        caption = m.group(1).strip()
        label_match = re.search(r"{#([^\}]+)}", caption)
        label = ""
        if label_match:
            raw_label = label_match.group(1)
            label = rf"\label{{{raw_label}}}"
            caption = caption[:label_match.start()].strip()
        path = m.group(2).strip()
        return (
            r"\begin{figure}[h]" + "\n"
            r"  \centering" + "\n"
            rf"  \includegraphics[width=1.0\textwidth]{{{path}}}" + "\n"
            rf"  \caption{{{caption}}}" + "\n"
            f"  {label}" + "\n"
            r"\end{figure}"
        )
    return re.sub(r"!\[(.*?)\]\((.*?)\)", repl, text)

def markdown_to_latex(text: str) -> str:
    text = convert_citations(text)
    text = convert_fig_refs(text)
    text = convert_inline_formatting(text)
    text, equations = convert_math(text)
    text = convert_sections(text)
    text = convert_lists(text)
    text = convert_images(text)
    text = restore_math(text, equations)
    return text

def convert_appendix(text: str) -> str:
    lines, out = text.splitlines(), []
    first_section = True
    for line in lines:
        m_sec = re.match(r"^##\s+(.*)$", line)
        if m_sec and first_section:
            out.append(rf"\section*{{Appendix: {m_sec.group(1)}}}")
            first_section = False
            continue
        m_sub = re.match(r"^###\s+(.*)$", line)
        if m_sub:
            out.append(rf"\subsection*{{{m_sub.group(1)}}}")
            continue
        out.append(line)
    return markdown_to_latex("\n".join(out))

# ----------------- Compile -----------------
def compile():
    abstract = sanitize_unicode(ABSTRACT_MD.read_text(encoding="utf-8").strip())
    body = sanitize_unicode(INDEX_MD.read_text(encoding="utf-8"))
    body = strip_title(body)
    body = strip_citation_section(body)

    abstract_tex = markdown_to_latex(abstract)
    body_tex = markdown_to_latex(body)

    appendix_tex = ""
    if APPENDIX_MD.exists():
        appendix_md = APPENDIX_MD.read_text(encoding="utf-8")
        appendix_tex = "\n\\newpage\n\\appendix\n\n" + convert_appendix(appendix_md)

    latex = PREAMBLE + "\n\\begin{abstract}\n" + abstract_tex + "\n\\end{abstract}\n\n" + body_tex + appendix_tex + POSTAMBLE
    OUT.write_text(latex, encoding="utf-8")
    print(f"Generated {OUT}")

    # Compilar PDF directamente en la raíz
    try:
        cmds = [["pdflatex", str(OUT)], ["bibtex", OUT.stem], ["pdflatex", str(OUT)], ["pdflatex", str(OUT)]]
        for cmd in cmds:
            subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"Generated PDF: {OUT.with_suffix('.pdf')}")
    except subprocess.CalledProcessError as e:
        print(f"Error during compilation: {e}")

if __name__ == "__main__":
    compile()