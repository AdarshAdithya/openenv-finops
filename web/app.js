const API_BASE = '';

let currentSessionId = null;
let currentStep = 0;
let isDone = false;

const grid = document.getElementById('resource-grid');
const totalCostEl = document.getElementById('total-cost');
const stepCounterEl = document.getElementById('step-counter');
const logContainer = document.getElementById('activity-log');
const btnStart = document.getElementById('btn-start');
const btnNop = document.getElementById('btn-nop');
const btnGrade = document.getElementById('btn-grade');
const modal = document.getElementById('grade-modal');
const statusChip = document.querySelector('.status-chip');

async function apiPost(endpoint, body = null) {
    const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(`${API_BASE}${endpoint}`, opts);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

async function apiGet(endpoint) {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

function log(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `log-entry ${type}`;
    el.textContent = `> ${msg}`;
    logContainer.prepend(el);
}

function setStatus(text, color) {
    statusChip.innerHTML = `<span class="dot" style="background:${color};box-shadow:0 0 8px ${color}"></span> ${text}`;
}

btnStart.addEventListener('click', async () => {
    try {
        logContainer.innerHTML = '';
        log('Initializing simulation...', 'info');
        setStatus('Loading...', 'var(--accent-cyan)');

        const data = await apiPost('/episodes');
        currentSessionId = data.episode_id;
        currentStep = 0;
        isDone = false;

        btnNop.disabled = false;
        btnGrade.disabled = true;
        btnStart.disabled = false;

        setStatus('Online', 'var(--dev-glow)');
        updateUI(data.initial_observation);
        log(`Session spawned [${currentSessionId.slice(0, 6)}]`);
    } catch (e) {
        log(`Failed: ${e.message}`, 'penalty');
        setStatus('Error', 'var(--prod-pulse)');
    }
});

btnNop.addEventListener('click', () => sendAction('nop', null));

document.getElementById('btn-close-modal').addEventListener('click', () => {
    modal.classList.add('hidden');
    if (isDone) {
        btnNop.disabled = true;
        btnGrade.disabled = true;
        btnStart.disabled = false;
        setStatus('Online', 'var(--dev-glow)');
    }
});

btnGrade.addEventListener('click', async () => {
    try {
        const data = await apiGet(`/episodes/${currentSessionId}/grade`);

        const easyBox = document.getElementById('easy-grade');
        const hardBox = document.getElementById('hard-grade');

        easyBox.innerHTML = `PASS: ${data.cost_optimisation.passed ? '✅' : '❌'}<br>SCORE: ${data.cost_optimisation.score.toFixed(3)}`;
        hardBox.innerHTML = `PASS: ${data.production_protection.passed ? '✅' : '❌'}<br>SCORE: ${data.production_protection.score.toFixed(3)}`;

        modal.classList.remove('hidden');
    } catch (e) {
        log(`Grading failed: ${e.message}`, 'penalty');
    }
});

async function sendAction(cmd, targetId) {
    if (isDone || !currentSessionId) return;

    try {
        const payload = {
            action: {
                type: cmd === 'nop' ? 'noop' : (cmd === 'terminate' ? 'terminate_service' : 'scale_down')
            }
        };
        if (targetId) payload.action.service = targetId;

        const data = await apiPost(`/episodes/${currentSessionId}/step`, payload);

        currentStep = data.step;
        isDone = data.done;

        const rtype = data.reward > 0 ? 'reward' : (data.reward < -1 ? 'penalty' : 'info');
        log(`${cmd.toUpperCase()} -> Reward: ${data.reward.toFixed(2)}`, rtype);

        updateUI(data.observation);

        if (isDone) {
            log('EPISODE COMPLETE — Click "Trigger Grading" to see results', 'info');
            btnNop.disabled = true;
            btnGrade.disabled = false;
            setStatus('Episode Done', 'var(--prod-pulse)');
        }
    } catch (e) {
        log(`Action Error: ${e.message}`, 'penalty');
    }
}

function updateUI(obs) {
    totalCostEl.textContent = obs.total_cost.toFixed(2);
    stepCounterEl.textContent = `${currentStep} / 10`;

    grid.innerHTML = '';

    obs.resources.forEach(res => {
        const utilPct = res.utilization * 100;

        const card = document.createElement('div');
        card.className = `resource-card ${res.is_prod ? 'prod' : 'dev'}`;
        card.id = `res-${res.id}`;

        card.innerHTML = `
            <div class="card-header">
                <h4>${res.id}</h4>
                <span class="badge ${res.is_prod ? 'prod' : 'dev'}">${res.is_prod ? 'PROD' : 'DEV'}</span>
            </div>
            <div class="card-stats">
                <div class="stat-block">
                    <span class="label">Type</span>
                    <span class="val">${res.type}</span>
                </div>
                <div class="stat-block" style="text-align: right;">
                    <span class="label">Cost</span>
                    <span class="val">$${res.cost.toFixed(2)}</span>
                </div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                    <span class="label" style="font-size:0.75rem; color:var(--text-main);">Utilization</span>
                    <span class="mono" style="font-size:0.75rem;">${utilPct.toFixed(0)}%</span>
                </div>
                <div class="utilization-bar">
                    <div class="fill ${utilPct > 70 ? 'high' : 'low'}" style="width: ${utilPct}%"></div>
                </div>
            </div>
            <div class="card-actions">
                <button class="btn secondary" onclick="sendAction('resize', '${res.id}')" ${res.is_prod ? 'disabled' : ''}>Scale Down</button>
                <button class="btn accent" onclick="triggerTerminate('${res.id}')">Terminate</button>
            </div>
        `;

        grid.appendChild(card);
    });
}

window.triggerTerminate = function(id) {
    const el = document.getElementById(`res-${id}`);
    if (el) {
        el.classList.add('terminated');
        setTimeout(() => sendAction('terminate', id), 300);
    } else {
        sendAction('terminate', id);
    }
}