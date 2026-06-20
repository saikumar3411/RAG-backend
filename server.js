require('dotenv').config()

const express = require('express')
const cors = require('cors')
const multer = require('multer')
const Anthropic = require('@anthropic-ai/sdk')
const fs = require('fs')
const pdf = require('pdf-parse')
const { ChromaClient } = require('chromadb')

const anthropic = new Anthropic()
const app = express()
const upload = multer({ dest: 'uploads/' })

app.use(cors())
app.use(express.json())

const chroma = new ChromaClient({ path: 'http://localhost:8000' })
let collection

async function init() {
    const { DefaultEmbeddingFunction } = await import('@chroma-core/default-embed')
    const embedder = new DefaultEmbeddingFunction()
    collection = await chroma.getOrCreateCollection({
        name: 'knowledge',
        embeddingFunction: embedder
    })
    console.log('ChromaDB collection ready')
}
init().catch(err => console.error('ChromaDB init failed:', err.message))

app.get('/', (req, res) => res.send('Backend is working'))

app.post('/upload', upload.single('file'), async (req, res) => {
    if (!req.file) return res.status(400).send('No file uploaded')

    try {
        const dataBuffer = fs.readFileSync(req.file.path)
        const data = await pdf(dataBuffer)
        fs.unlinkSync(req.file.path)

        const chunks = data.text
            .split('\n\n')
            .map(c => c.trim())
            .filter(c => c.length > 50)

        const ids = chunks.map((_, i) => `${req.file.originalname}-${Date.now()}-${i}`)
        const metadatas = chunks.map(() => ({ source: req.file.originalname }))

        await collection.add({ ids, documents: chunks, metadatas })

        res.json({ message: `Indexed ${chunks.length} chunks from "${req.file.originalname}"` })
    } catch (err) {
        console.error(err)
        res.status(500).send('Failed to parse PDF')
    }
})

app.post('/chat', async (req, res) => {
    const { question } = req.body
    if (!question) return res.status(400).send('No question provided')

    try {
        const results = await collection.query({ queryTexts: [question], nResults: 3 })
        const context = results.documents[0].join('\n\n')

        const response = await anthropic.messages.create({
            model: 'claude-opus-4-8',
            max_tokens: 1024,
            messages: [{
                role: 'user',
                content: `Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."\n\nContext:\n${context}\n\nQuestion: ${question}`
            }]
        })

        res.json({ answer: response.content[0].text })
    } catch (err) {
        console.error(err)
        res.status(500).send('Failed to generate answer')
    }
})

const PORT = process.env.PORT || 3000
app.listen(PORT, () => console.log(`Server listening on PORT ${PORT}`))
