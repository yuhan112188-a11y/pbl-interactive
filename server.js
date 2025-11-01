
import 'dotenv/config'
import express from 'express'
import fetch from 'node-fetch'
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
app.use(express.json({ limit: '1mb' }))
app.use(express.static(path.join(__dirname, 'public')))

// Load cards
const CARDS = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'case_cards.json'), 'utf8'))

// Health check
app.get('/health', (req,res)=>res.json({ok:true}))

// Embedding function
async function getEmbedding(text){
  const url = `${process.env.EMBED_BASE_URL || 'https://api.deepseek.com/v1'}/embeddings`
  const res = await fetch(url, {
    method:'POST',
    headers:{
      'Authorization': `Bearer ${process.env.DS_API_KEY}`,
      'Content-Type':'application/json'
    },
    body: JSON.stringify({
      input: text,
      model: process.env.EMBED_MODEL || 'deepseek-embedding-2'
    })
  })
  if (!res.ok){
    const body = await res.text()
    throw new Error(`Embedding API error ${res.status}: ${body}`)
  }
  const data = await res.json()
  return data.data?.[0]?.embedding || data.embedding || data.vector
}

function norm(v){ return Math.sqrt(v.reduce((s,x)=>s + x*x, 0)) || 1 }
function dot(a,b){ let s=0; for (let i=0;i<a.length;i++) s+=a[i]*b[i]; return s }

let CARD_VECS = []

async function buildIndex(){
  CARD_VECS = []
  for (const c of CARDS){
    const text = `${c.title} ${c.content} ${(c.synonyms||[]).join(' ')}`
    const vec = await getEmbedding(text)
    CARD_VECS.push({ id: c.id, case_id: c.case_id, vec, n: norm(vec) })
  }
  console.log(`Indexed ${CARD_VECS.length} cards.`)
}

// Bootstrap endpoint: returns the initial chief complaint for the case
app.get('/bootstrap', (req,res)=>{
  const case_id = String(req.query.case_id || '1')
  const chief = CARDS.find(c => c.case_id === case_id && c.initial)
  if (!chief) return res.status(404).json({error:'No initial card'})
  res.json({ initial: { id: chief.id, title: chief.title, content: chief.content } })
})

// Ask endpoint
app.post('/ask', async (req,res)=>{
  try{
    const { case_id, question, revealed_ids = [] } = req.body || {}
    if (!case_id || !question) return res.status(400).json({ error: 'case_id & question required' })

    const qVec = await getEmbedding(question)
    const qn = norm(qVec)

    const threshold = Number(process.env.SIM_THRESHOLD || 0.40)
    const K = Number(process.env.TOP_K || 1)

    const candidates = CARD_VECS.filter(v => v.case_id === String(case_id) && !revealed_ids.includes(v.id))

    const scored = candidates.map(v => ({
      id: v.id,
      score: dot(qVec, v.vec) / (qn * v.n)
    })).sort((a,b) => b.score - a.score)

    const hits = []
    for (const sc of scored){
      if (hits.length >= K) break
      if (sc.score >= threshold) hits.push(sc)
    }

    if (hits.length === 0){
      return res.json({ reply_blocks: [], newly_revealed_ids: [], nohit: true })
    }

    const byId = new Map(CARDS.map(c => [c.id, c]))
    const reply_blocks = hits.map(h => {
      const c = byId.get(h.id)
      return { id: c.id, title: c.title, content: c.content }
    })
    const newly_revealed_ids = reply_blocks.map(b => b.id)

    res.json({ reply_blocks, newly_revealed_ids, nohit: false })
  }catch(e){
    console.error(e)
    res.status(500).json({ error: String(e) })
  }
})

// Serve SPA
app.get('*', (req,res)=>{
  res.sendFile(path.join(__dirname, 'public', 'index.html'))
})

buildIndex().then(()=>{
  const port = Number(process.env.PORT || 8787)
  app.listen(port, ()=>console.log(`PBL server running on :${port}`))
})
