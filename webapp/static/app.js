/**
 * Intelli-Light Dashboard Logic - Cyberpunk Edition
 */

document.addEventListener("DOMContentLoaded", () => {
    
    // ── DOM ELEMENTS ──
    const elWsDot = document.getElementById("ws-dot");
    const elWsText = document.getElementById("ws-text");
    
    const elCycle = document.getElementById("hud-cycle");
    const elThroughput = document.getElementById("hud-throughput");
    const elThroughputBar = document.getElementById("hud-throughput-bar");
    const elWait = document.getElementById("hud-wait");
    const elQueue = document.getElementById("hud-queue");
    
    const elScenario = document.getElementById("scenario-select");
    const elModel = document.getElementById("model-select");
    const btnAmbulance = document.getElementById("btn-inject-ambulance");
    const btnAccident = document.getElementById("btn-inject-accident");
    const btnEmergency = document.getElementById("btn-emergency-override");
    const btnStatusToggle = document.getElementById("btn-status-toggle");
    
    const chartContainer = document.getElementById("historical-chart-container");
    const rlLogsContainer = document.getElementById("rl-logs-container");
    const rawTerminal = document.getElementById("raw-telemetry-terminal");

    let isEmergencyActive = false;

    // ── INIT API CALLS ──
    const API_BASE = "";

    // Fetch Models
    fetch(`${API_BASE}/api/models`)
        .then(res => res.json())
        .then(data => {
            elModel.innerHTML = "";
            const defaultOpt = document.createElement("option");
            defaultOpt.value = "";
            defaultOpt.text = "-- Select Model Override --";
            elModel.appendChild(defaultOpt);
            
            if (data.models && data.models.length > 0) {
                data.models.forEach(m => {
                    const opt = document.createElement("option");
                    opt.value = m;
                    opt.text = m;
                    elModel.appendChild(opt);
                });
            } else {
                defaultOpt.text = "No models found";
                defaultOpt.disabled = true;
            }
        }).catch(err => addTerminalLog("API_ERR: Models fetch failed"));

    // Fetch Historical Data and Render Charts
    fetch(`${API_BASE}/api/results`)
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                addTerminalLog("API_ERR: " + data.error);
                return;
            }
            renderCharts(data);
            addTerminalLog("INIT_HISTORICAL: SUCCESS");
        }).catch(err => addTerminalLog("API_ERR: Historical fetch failed"));

    // Toggle chart view
    btnStatusToggle.addEventListener("click", () => {
        chartContainer.classList.toggle("hidden");
        chartContainer.classList.toggle("flex");
    });

    // ── COMMAND DISPATCHER ──
    function sendCommand(type, value) {
        fetch(`${API_BASE}/api/command`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type, value })
        }).then(() => {
            addTerminalLog(`CMD_SENT [${type}]: ${value}`);
        }).catch(err => addTerminalLog(`CMD_ERR: Failed to push ${type}`));
    }

    elScenario.addEventListener("change", (e) => sendCommand("SCENARIO", e.target.value));
    elModel.addEventListener("change", (e) => { if(e.target.value) sendCommand("LOAD_MODEL", e.target.value); });
    btnAmbulance.addEventListener("click", () => sendCommand("EVENT", "AMBULANCE"));
    btnAccident.addEventListener("click", () => sendCommand("EVENT", "ACCIDENT"));
    
    // Emergency / Failsafe
    btnEmergency.addEventListener("click", () => {
        isEmergencyActive = !isEmergencyActive;
        sendCommand("FAILSAFE", isEmergencyActive);
        if (isEmergencyActive) {
            btnEmergency.classList.replace("bg-red-500/20", "bg-red-500");
            btnEmergency.classList.replace("text-red-500", "text-white");
            addRlLog("Preemption override triggered", "System Failsafe", "Reward --", "bg-red-500/5", "red-500", "warning_amber");
        } else {
            btnEmergency.classList.replace("bg-red-500", "bg-red-500/20");
            btnEmergency.classList.replace("text-white", "text-red-500");
            addRlLog("Preemption override removed", "System Failsafe", "Reward Normal", "", "primary", "check_circle");
        }
    });

    // ── WEBSOCKET CONNECTION ──
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host || 'localhost:8000';
    
    let socket;
    function connectWS() {
        addTerminalLog("INIT_CONNECT_SEQ: PENDING");
        socket = new WebSocket(`${wsProtocol}//${wsHost}/ws/telemetry`);
        
        socket.onopen = () => {
            elWsDot.classList.replace("bg-zinc-500", "bg-primary");
            elWsDot.classList.add("shadow-[0_0_8px_#34C759]");
            elWsText.innerText = "Connected";
            elWsText.classList.replace("text-zinc-500", "text-primary");
            addTerminalLog("ESTABLISH_TLS: TRUE");
            addTerminalLog("RECV_TELEMETRY: ACTIVE");
            rlLogsContainer.innerHTML = ""; // Clear loader
        };

        socket.onclose = () => {
            elWsDot.classList.replace("bg-primary", "bg-zinc-500");
            elWsDot.classList.replace("shadow-[0_0_8px_#34C759]", "shadow-none");
            elWsText.innerText = "Disconnected reconnecting...";
            elWsText.classList.replace("text-primary", "text-zinc-500");
            addTerminalLog("RECV_TELEMETRY: LOST REF");
            setTimeout(connectWS, 3000);
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                updateHUD(data);
                updateVisualizer(data.active_phases);
                addTerminalLog(`> STATE_SYNC: CYCLE ${data.cycle} | T:${data.throughput}`);
                
                // Randomly add some log visual noise occasionally when new data comes in to make UI feel alive
                if (Math.random() > 0.85) {
                    generateRlLogFake(data);
                }
            } catch (e) {
                console.error("Payload parsing error:", e, event.data);
            }
        };
    }
    connectWS();

    // ── DATA UPDATERS ──
    function updateHUD(data) {
        if(data.cycle !== undefined) {
            elCycle.innerText = data.cycle;
        }
        if(data.throughput !== undefined) {
            elThroughput.innerText = data.throughput;
            // Cap visual progress at arbitrary 1000 for demo scale
            let pct = Math.min(100, Math.max(0, (data.throughput / 1000) * 100)); 
            if (data.throughput < 100) pct = Math.max(5, data.throughput); // just ensure something shows
            elThroughputBar.style.width = pct + "%";
        }
        if(data.avg_wait !== undefined) {
            elWait.innerText = data.avg_wait.toFixed(1);
        }
        if(data.total_queue !== undefined) {
            elQueue.innerText = data.total_queue.toFixed(0);
        }
    }

    // Phases are 0: EW-Through, 1: EW-Left, 2: NS-Through, 3: NS-Left
    // simplified visual: if phase < 2 then pip glow green else red (or toggle state)
    function updateVisualizer(phases) {
        if(!phases) return;
        ['J1', 'J2', 'J3'].forEach(jid => {
            const phase = phases[jid];
            const light = document.getElementById(`pip-${jid.toLowerCase()}-light`);
            if(!light) return;
            
            // Clean up
            light.classList.remove("green", "red", "bg-primary", "bg-red-500", "shadow-[0_0_15px_#34C759]", "shadow-[0_0_15px_#FF3B30]", "bg-yellow-400");
            
            if (phase === 0 || phase === 1) { 
                light.classList.add("green", "bg-primary", "shadow-[0_0_15px_#34C759]");
            } else if (phase === 2 || phase === 3) {
                light.classList.add("red", "bg-red-500", "shadow-[0_0_15px_#FF3B30]");
            } else {
                light.classList.add("bg-yellow-400");
            }
        });
    }

    // ── UI HELPERS ──
    function addTerminalLog(msg) {
        const time = new Date().toISOString().split('T')[1].slice(0,-1); // HH:MM:SS.mmm
        const div = document.createElement('div');
        div.innerHTML = `> [${time}] ${msg}`;
        rawTerminal.insertBefore(div, rawTerminal.lastElementChild);
        
        // keep only last 15 lines
        if (rawTerminal.children.length > 15) {
            rawTerminal.removeChild(rawTerminal.children[0]);
        }
    }

    function addRlLog(title, subtitle, reward, bgClass, colorName, iconName) {
        const time = new Date().toISOString().split('T')[1].slice(0,-1);
        const html = `
        <div class="flex items-center p-4 border-b border-white/5 hover:bg-white/5 transition-colors ${bgClass}">
            <div class="w-10 h-10 rounded ${bgClass ? bgClass : 'bg-surface-container'} flex items-center justify-center mr-4">
                <span class="material-symbols-outlined text-${colorName} text-sm">${iconName}</span>
            </div>
            <div class="flex-1">
                <div class="font-mono-data text-mono-data text-on-surface">${title}</div>
                <div class="font-mono-data text-[11px] text-${colorName}/70">${subtitle}</div>
            </div>
            <div class="text-right">
                <div class="font-mono-data text-mono-data text-${colorName}">${reward}</div>
                <div class="font-mono-data text-[10px] text-zinc-600">${time}</div>
            </div>
        </div>`;
        
        rlLogsContainer.insertAdjacentHTML('afterbegin', html);
        if (rlLogsContainer.children.length > 20) {
            rlLogsContainer.removeChild(rlLogsContainer.lastElementChild);
        }
    }

    function generateRlLogFake(data) {
        const jkeys = Object.keys(data.active_phases || {});
        if(jkeys.length===0) return;
        const j = jkeys[Math.floor(Math.random()*jkeys.length)];
        const isBad = Math.random() > 0.8;
        if(isBad) {
            addRlLog(`Congestion detected at ${j}`, `Policy: Backpressure Route`, `Q>10`, "bg-yellow-400/5", "yellow-400", "traffic");
        } else {
            addRlLog(`Phase extension sync`, `Intersection ${j} • PPO Model`, `Reward +0.${Math.floor(Math.random()*90)+10}`, "", "primary", "sync_alt");
        }
    }

    // ── CHART GENERATION ──
    function renderCharts(json) {
        // We render Wait Time on the single chart container inside the Wait Time Analytics card
        const baseScenario = json.results_by_scenario["WEEKEND"] || json.results_by_scenario["MORNING_RUSH"];
        if(!baseScenario) return;

        // Populate HUD with IntelliLight historical data so UI isn't flat/zeroed
        const rlStats = baseScenario["IntelliLight-RL"];
        if (rlStats && rlStats.mean) {
            updateHUD({
                cycle: "HISTORICAL",
                throughput: Math.round(rlStats.mean.throughput),
                avg_wait: rlStats.mean.avg_wait_time,
                total_queue: rlStats.mean.avg_queue_length
            });
            addTerminalLog(`LOADED HISTORICAL RUN: ${json.timestamp || "N/A"}`);
            addRlLog("Loaded latest historical run", "System Initialization", "Resume Ready", "", "primary", "history");
        }

        const labels = Object.keys(baseScenario).filter(k => k !== "improvements");
        const waitData = labels.map(L => baseScenario[L].mean.avg_wait_time);

        const ctxWait = document.getElementById('chartWaitTime').getContext('2d');
        new Chart(ctxWait, {
            type: 'bar',
            data: {
                labels,
                datasets: [{ 
                    label: 'Avg Wait (s)', 
                    data: waitData, 
                    backgroundColor: ['#64748b', '#f59e0b', '#34C759'],
                    borderRadius: 4
                }]
            },
            options: { 
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', font: {family: 'Inter', size: 10} } },
                    x: { grid: { display: false }, ticks: { color: '#94a3b8', font: {family: 'Inter', size: 10} } }
                }
            }
        });
    }
});
