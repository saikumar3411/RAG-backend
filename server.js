require('dotenv').config()

const express = require('express')
const cors = require('cors')
const multer = require('multer')
const Anthropic = require('@anthropic-ai/sdk')
const fs = require('fs')
const pdf = require('pdf-parse')
const { pipeline } = require('@xenova/transformers')

const anthropic = new Anthropic()
const app = express()
const upload = multer({ dest: 'uploads/' })

app.use(cors())
app.use(express.json())

const vectorStore = []

let extractor
async function init() {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2')
    console.log('Embedding model ready')
}
init()

async function getEmbedding(text) {
    const output = await extractor(text, { pooling: 'mean', normalize: true })
    return Array.from(output.data)
}

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0
    for (let i = 0; i < a.length; i++) {
        dot   += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

function queryStore(queryVector, nResults = 3) {
    return vectorStore
        .map(entry => ({ ...entry, score: cosineSimilarity(queryVector, entry.embedding) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, nResults)
        .map(entry => entry.document)
}

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

        for (let i = 0; i < chunks.length; i++) {
            const embedding = await getEmbedding(chunks[i])
            vectorStore.push({
                id: `${req.file.originalname}-${Date.now()}-${i}`,
                embedding,
                document: chunks[i],
                source: req.file.originalname
            })
        }

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
        const questionVector = await getEmbedding(question)
        const topChunks = queryStore(questionVector, 3)
        const context = topChunks.join('\n\n')

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
