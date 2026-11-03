import io, re, math
from collections import Counter

def words(text):
    return re.findall(r"[a-zA-Z0-9']+", (text or "").lower())

def load_documents(name, data):
    if name.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader=PdfReader(io.BytesIO(data))
            raw="\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e: raise RuntimeError(f"PDF reading failed: {e}")
    else: raw=data.decode("utf-8",errors="ignore")
    paras=[x.strip() for x in re.split(r"\n\s*\n",raw) if x.strip()]
    chunks=[]; current=""
    for para in paras:
        if len(current)+len(para)>850 and current:
            chunks.append(current.strip()); current=""
        current += para+"\n\n"
    if current.strip(): chunks.append(current.strip())
    if not chunks and raw.strip(): chunks=[raw[:850]]
    return [{"text":c,"source":name,"chunk_id":i} for i,c in enumerate(chunks)]

def build_index(docs):
    corpus=[words(d["text"]) for d in docs]
    df=Counter()
    for t in corpus: df.update(set(t))
    n=max(1,len(corpus)); result=[]
    for doc,t in zip(docs,corpus):
        tf=Counter(t)
        vec={w:(1+math.log(v))*(math.log((n+1)/(df[w]+1))+1) for w,v in tf.items()}
        norm=math.sqrt(sum(v*v for v in vec.values())) or 1
        result.append({**doc,"vec":{w:v/norm for w,v in vec.items()}})
    return result

def retrieve(query,index,top_k=4):
    q=Counter(words(query))
    qv={w:1+math.log(v) for w,v in q.items()}
    norm=math.sqrt(sum(v*v for v in qv.values())) or 1
    qv={w:v/norm for w,v in qv.items()}
    scored=[]
    for item in index:
        score=sum(v*item["vec"].get(w,0) for w,v in qv.items())
        scored.append({"text":item["text"],"source":item["source"],"chunk_id":item["chunk_id"],"score":score})
    return sorted(scored,key=lambda x:x["score"],reverse=True)[:top_k]

def ratio(a,b):
    return len(a & b)/max(1,len(b))

def evaluate_answer(question,answer,contexts,reference=""):
    a=set(words(answer)); q=set(words(question))
    context_words=set(words(" ".join(c["text"] for c in contexts)))
    # Remove common glue words to make lexical indicators less noisy.
    stop={"the","a","an","is","are","was","were","to","of","and","or","in","on","for","it","this","that","with","as","by","be","from","at","if","per"}
    ac=a-stop; cc=context_words-stop; qc=q-stop
    grounded=ratio(ac,cc)
    relevance=ratio(ac,qc) if qc else 0.0
    # Count retrieved chunks that share meaningful query terms.
    precision=sum(bool((set(words(c["text"]))-stop)&qc) for c in contexts)/max(1,len(contexts))
    ref=set(words(reference))-stop
    coverage=ratio(ac,ref) if ref else None
    unsupported=1-grounded
    return {
        "groundedness":round(grounded,3),
        "answer_relevance":round(relevance,3),
        "retrieval_precision":round(precision,3),
        "reference_coverage":None if coverage is None else round(coverage,3),
        "unsupported_word_risk":round(unsupported,3),
        "explanation":"These are explainable lexical indicators, not semantic truth scores. Word overlap can miss paraphrases and can reward copied wording. Review the answer and evidence manually."
    }

def run_batch(df,index,top_k):
    required={"question","answer"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain question and answer columns.")
    rows=[]
    for _,row in df.iterrows():
        q=str(row["question"]); a=str(row["answer"])
        ref=str(row.get("reference_answer","") or "")
        ctx=retrieve(q,index,top_k)
        m=evaluate_answer(q,a,ctx,ref)
        rows.append({"question":q,"answer":a,**{k:v for k,v in m.items() if k!="explanation"}})
    return rows
