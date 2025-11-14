const transcriptEl = document.getElementById('transcript')
const stateEl = document.getElementById('state')
const apiResultEl = document.getElementById('api-result')
const inputEl = document.getElementById('input')
const formEl = document.getElementById('chat-form')
const systemEl = document.getElementById('system')
const sendBtn = document.getElementById('send')
const scriptedEl = document.getElementById('scripted')
const kbFilesEl = document.getElementById('kb-files')
const uploadKbBtn = document.getElementById('upload-kb')
const kbQueryEl = document.getElementById('kb-query')
const searchKbBtn = document.getElementById('search-kb')
const reloadScriptBtn = document.getElementById('reload-script')
const submitCustomerBtn = document.getElementById('submit-customer')
const selectedItemsEl = document.getElementById('selected-items')
const firstNameEl = document.getElementById('first-name')
const lastNameEl = document.getElementById('last-name')
const phoneEl = document.getElementById('phone')
const emailEl = document.getElementById('email')
const deliveryAddressEl = document.getElementById('delivery-address')
const customerResultEl = document.getElementById('customer-result')
const pingBtn = document.getElementById('ping')
const pingResultEl = document.getElementById('ping-result')

let messages = []
let companyId = 'default'
let turn = 0

function render() {
  transcriptEl.innerHTML = ''
  for (const m of messages) {
    const div = document.createElement('div')
    div.className = `msg ${m.role}`
    const role = document.createElement('div')
    role.className = 'role'
    role.textContent = m.role
    const bubble = document.createElement('div')
    bubble.className = 'bubble'
    bubble.textContent = m.content
    div.appendChild(role)
    div.appendChild(bubble)
    transcriptEl.appendChild(div)
  }
}

function ensureSystemPrompt() {
  const sys = systemEl.value.trim()
  const existing = messages.find((m) => m.role === 'system')
  if (!existing && sys) messages.unshift({ role: 'system', content: sys })
  else if (existing && existing.content !== sys) existing.content = sys
}

async function loadState() {
  try {
    const res = await fetch(`/api/state`)
    if (res.ok) {
      const st = await res.json()
      renderState(st)
      companyId = 'default'
    }
  } catch {}
}

function renderState(st) {
  const lines = []
  lines.push(`State: ${st.current_state}`)
  const slots = st.slots || {}
  const keys = Object.keys(slots)
  if (keys.length) {
    lines.push('Slots:')
    for (const k of keys) lines.push(`- ${k}: ${slots[k]}`)
  } else {
    lines.push('Slots: (none)')
  }
  stateEl.textContent = lines.join('\n')
}


async function reloadScript() {
  try {
    const res = await fetch(`/api/state/script/reload`, { method: 'POST' })
    const data = await res.json()
    if (data && data.ok) {
      await loadState()
      alert('Script reloaded for this conversation.')
    } else {
      alert('Failed to reload script')
    }
  } catch (e) {
    alert(String(e))
  }
}

async function uploadKB() {
  try {
    const fd = new FormData()
    const files = kbFilesEl.files
    for (let i = 0; i < files.length; i++) fd.append('files', files[i])
    const res = await fetch(`/api/kb/upload`, {
      method: 'POST',
      body: fd,
    })
    const data = await res.json()
    alert(`KB uploaded: ${data.chunks_added} chunks`)
  } catch (e) {
    alert(String(e))
  }
}

async function searchKB() {
  try {
    const query = kbQueryEl.value.trim()
    if (!query) return
    const persona = systemEl.value.trim()
    const res = await fetch(`/api/kb/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, persona, tone: '', org_name: '' }),
    })
    const data = await res.json()
    if (data.reply) {
      messages.push({ role: 'assistant', content: data.reply })
      render()
    } else {
      alert('No reply from KB')
    }
    if (data.kb && Array.isArray(data.kb.sources)) {
      const lines = ['KB Sources:']
      for (const s of data.kb.sources) {
        const score = typeof s.score === 'number' ? s.score.toFixed(4) : s.score
        lines.push(`${s.source_path || '(unknown)'} (score=${score})`)
        lines.push((s.preview || '').replace(/\s+/g, ' ').slice(0, 160))
      }
      document.getElementById('kb-sources').textContent = lines.join('\n')
    }
  } catch (e) {
    alert(String(e))
  }
}

async function submitCustomer() {
  try {
    const items = selectedItemsEl.value.split(',').map((s) => s.trim()).filter((s) => s.length)
    const payload = {
      selected_items: items,
      customer_details: {
        first_name: firstNameEl.value.trim(),
        last_name: lastNameEl.value.trim(),
        phone: phoneEl.value.trim(),
        email: emailEl.value.trim(),
        delivery_address: deliveryAddressEl.value.trim(),
      },
    }
    const res = await fetch('/api/customer/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    const data = await res.json()
    customerResultEl.textContent = JSON.stringify(data, null, 2)
  } catch (e) {
    customerResultEl.textContent = String(e)
  }
}

async function ping() {
  try {
    const res = await fetch('/api/ping')
    const data = await res.json()
    pingResultEl.textContent = JSON.stringify(data, null, 2)
  } catch (e) {
    pingResultEl.textContent = String(e)
  }
}

formEl.addEventListener('submit', async (e) => {
  e.preventDefault()
  const text = inputEl.value.trim()
  if (!text) return
  inputEl.value = ''
  const useScripted = scriptedEl.checked
  if (!useScripted) {
    ensureSystemPrompt()
    messages.push({ role: 'user', content: text })
    render()
  }
  sendBtn.disabled = true
  try {
    let data
    if (useScripted) {
      messages.push({ role: 'user', content: text })
      render()
      const res = await fetch(`/api/scripted_chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text, turn, history: messages.slice(-8) }),
      })
      data = await res.json()
      if (data.reply) messages.push({ role: 'assistant', content: data.reply })
      if (data.kb && Array.isArray(data.kb.sources)) {
        const lines = ['KB Sources:']
        for (const s of data.kb.sources) {
          const score = typeof s.score === 'number' ? s.score.toFixed(4) : s.score
          lines.push(`${s.source_path || '(unknown)'} (score=${score})`)
          lines.push((s.preview || '').replace(/\s+/g, ' ').slice(0, 160))
        }
        document.getElementById('kb-sources').textContent = lines.join('\n')
      }
      if (data.api) {
        apiResultEl.textContent = JSON.stringify(data.api, null, 2)
      }
      render()
      turn++
    } else {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      })
      data = await res.json()
      if (data && data.reply) {
        messages.push({ role: 'assistant', content: data.reply })
        render()
      } else {
        alert('No reply')
      }
    }
  } catch (err) {
    alert(String(err))
  } finally {
    sendBtn.disabled = false
  }
})



uploadKbBtn.addEventListener('click', uploadKB)
searchKbBtn.addEventListener('click', searchKB)
reloadScriptBtn.addEventListener('click', reloadScript)
submitCustomerBtn.addEventListener('click', submitCustomer)
pingBtn.addEventListener('click', ping)

render()
loadState()
