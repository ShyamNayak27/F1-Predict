/**
 * F1 Predict — app.js
 * Loads race_results.json and drives the entire dashboard.
 * Fan-first: rankings and win% up front, model details in Nerd Zone.
 */

'use strict';

// ════════════════════════════════════════════════════════════════
// STATE
// ════════════════════════════════════════════════════════════════
const state = {
  data: null,           // full JSON payload
  selectedRound: null,  // currently viewed race round number
  mode: 'pre',          // 'pre' | 'post'
  sortMode: 'predicted',// 'predicted' | 'actual'
  nerdOpen: false,
  nerdTab: 'pipeline',  // currently selected model details tab
  featTab: 'reg',       // 'reg' | 'win' | 'dnf'
  charts: {
    trend: null,
    feat: null,
  },
};

// ════════════════════════════════════════════════════════════════
// BOOT
// ════════════════════════════════════════════════════════════════
async function boot() {
  try {
    const resp = await fetch('race_results.json?t=' + Date.now());
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    state.data = await resp.json();
    hideEl('loadingState');
    showEl('dashboard');
    buildRaceSelector();
    // Default to last race with actuals (post-race view for that race)
    const racesWithActuals = state.data.races.filter(r => r.has_actuals);
    const defaultRound = racesWithActuals.length > 0
      ? racesWithActuals[racesWithActuals.length - 1].round
      : state.data.races[0].round;
    selectRound(defaultRound);
  } catch (e) {
    hideEl('loadingState');
    showEl('errorState');
    console.error('Failed to load race_results.json:', e);
  }
}

// ════════════════════════════════════════════════════════════════
// RACE SELECTOR (SIDEBAR)
// ════════════════════════════════════════════════════════════════
function buildRaceSelector() {
  const listEl = el('raceList');
  listEl.innerHTML = '';

  state.data.races.forEach(race => {
    const btn = document.createElement('button');
    btn.className = 'race-menu-item';
    btn.id = `pill-${race.round}`;
    btn.setAttribute('aria-label', race.race_name);
    btn.onclick = () => selectRound(race.round);

    const dotClass = !race.has_actuals ? 'no-data'
      : race.metrics?.winner_correct ? 'hit' : 'miss';

    btn.innerHTML = `
      <span class="race-menu-round">R${race.round}</span>
      <span class="race-menu-code">${raceCode(race.circuit_id)}</span>
      <span class="race-menu-dot ${dotClass}"></span>
    `;
    listEl.appendChild(btn);
  });

  // "NEXT" menu item placeholder
  const nextItem = document.createElement('div');
  nextItem.className = 'race-menu-item next-race';
  nextItem.innerHTML = `
    <span class="race-menu-round">NEXT</span>
    <span class="race-menu-code">TBD</span>
    <span class="race-menu-dot no-data"></span>
  `;
  listEl.appendChild(nextItem);
}

// ════════════════════════════════════════════════════════════════
// ROUND SELECTION
// ════════════════════════════════════════════════════════════════
function selectRound(roundNum) {
  state.selectedRound = roundNum;
  const race = state.data.races.find(r => r.round === roundNum);
  if (!race) return;

  // Update active nav item
  document.querySelectorAll('.race-menu-item').forEach(p => p.classList.remove('active'));
  const activeItem = el(`pill-${roundNum}`);
  if (activeItem) {
    activeItem.classList.add('active');
    activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // If race has actuals and we're in pre mode, keep pre (let user switch)
  renderRace(race);
}

// ════════════════════════════════════════════════════════════════
// RENDER RACE
// ════════════════════════════════════════════════════════════════
function renderRace(race) {
  renderHero(race);
  renderPodium(race);
  renderDNFWatch(race);
  renderGrid(race);
  renderTrackRecord();
  renderTrendChart();
  renderSharePreview(race);
  renderNerdZone(); // only builds once; feature tabs handle the rest
}

// ── HERO ─────────────────────────────────────────────────────
function renderHero(race) {
  const isUpcoming = !race.has_actuals;
  const statusBadge = isUpcoming ? ' <span class="upcoming-badge">UPCOMING</span>' : '';
  el('activeRaceTitle').innerHTML = `Round ${race.round} &mdash; ${race.race_name}${statusBadge}`;

  // Sync Nerd Zone toggle sidebar button states
  const sidebarBtn = el('sidebarNerdToggle');
  if (sidebarBtn) {
    sidebarBtn.setAttribute('aria-expanded', state.nerdOpen);
    const arrow = el('sidebarNerdArrow');
    if (arrow) arrow.classList.toggle('open', state.nerdOpen);
  }
}

// ── PODIUM & FORECAST CARDS ──────────────────────────────────
function renderPodium(race) {
  const container = el('kpiGrid');
  container.innerHTML = '';

  const topThree = [...race.drivers]
    .sort((a, b) => a.predicted_pos - b.predicted_pos)
    .slice(0, 3);

  const posLabels = ['P1 — OUR PICK', 'P2 — OUR PICK', 'P3 — OUR PICK'];
  const posKeys   = ['p1', 'p2', 'p3'];

  // 1. Render Podium Cards
  topThree.forEach((driver, i) => {
    const card = document.createElement('div');
    card.className = `kpi-card podium-card ${posKeys[i]}`;
    card.style.setProperty('--team-color', driver.team_color);

    const winPct = Math.round(driver.win_prob * 100);
    const circumference = 2 * Math.PI * 18;
    const offset = circumference - (winPct / 100) * circumference;

    // Post-race accuracy badge
    let accuracyHTML = '';
    if (race.has_actuals && state.mode === 'post' && driver.actual_pos !== null) {
      const err = Math.abs(driver.error);
      const cls = err === 0 ? 'hit' : err <= 3 ? 'close' : 'miss';
      const label = err === 0
        ? 'Nailed it!'
        : err <= 2 ? `Diff: ${driver.error > 0 ? '+' : ''}${driver.error}`
        : `Actually P${driver.actual_pos}`;
      accuracyHTML = `<div class="podium-accuracy ${cls}">${label}</div>`;
    }

    card.innerHTML = `
      <div class="podium-pos-badge">${posLabels[i]}</div>
      <div class="podium-driver-code">${driver.code}</div>
      <div class="podium-driver-name">${driver.full_name}</div>
      <div class="podium-team-name" style="color:${driver.team_color}">${teamName(driver.constructor_id)}</div>
      <div class="podium-win-wrap">
        <div class="podium-win-ring">
          <svg viewBox="0 0 44 44" width="44" height="44">
            <circle class="ring-bg" cx="22" cy="22" r="18"/>
            <circle class="ring-fill" cx="22" cy="22" r="18"
              stroke-dasharray="${circumference}"
              stroke-dashoffset="${offset}"/>
          </svg>
        </div>
        <div class="podium-win-text">
          <span class="podium-win-pct">${winPct}%</span>
          <span class="podium-win-label">win chance</span>
        </div>
      </div>
      ${accuracyHTML}
    `;
    container.appendChild(card);
  });

  // 2. Render Forecast Card
  const forecastCard = document.createElement('div');
  forecastCard.className = 'kpi-card forecast-card';

  const scProb = Math.round(race.sc_prob * 100);
  const rainProb = Math.round(race.rain_prob * 100);

  forecastCard.innerHTML = `
    <div class="podium-pos-badge">RACE FORECAST</div>
    
    <div class="forecast-metric">
      <div class="forecast-label">Safety Car Probability</div>
      <div class="forecast-value-row">
        <span class="forecast-value">${scProb}%</span>
        <div class="forecast-mini-bar"><div class="fill" style="width: ${scProb}%"></div></div>
      </div>
    </div>
    
    <div class="forecast-metric" style="margin-top: 14px;">
      <div class="forecast-label">Rain Probability</div>
      <div class="forecast-value-row">
        <span class="forecast-value">${rainProb}%</span>
        <div class="forecast-mini-bar"><div class="fill" style="width: ${rainProb}%"></div></div>
      </div>
    </div>
  `;
  container.appendChild(forecastCard);
}

// ── DNF WATCH ────────────────────────────────────────────────
function renderDNFWatch(race) {
  const highRisk = [...race.drivers]
    .filter(d => d.dnf_risk >= 0.07)
    .sort((a, b) => b.dnf_risk - a.dnf_risk)
    .slice(0, 5);

  const watchEl = el('dnfWatch');
  if (highRisk.length === 0) {
    watchEl.classList.add('hidden');
    return;
  }
  watchEl.classList.remove('hidden');

  el('dnfDrivers').innerHTML = highRisk.map(d => `
    <div class="dnf-driver-chip">
      <span style="color:${d.team_color}">●</span>
      <span>${d.code}</span>
      <span class="dnf-pct">${Math.round(d.dnf_risk * 100)}%</span>
    </div>
  `).join('');
}

// ── GRID TABLE ───────────────────────────────────────────────
function renderGrid(race) {
  const showActual = race.has_actuals && state.mode === 'post';

  // Show/hide actual columns
  el('thActual').classList.toggle('hidden', !showActual);
  el('thAcc').classList.toggle('hidden', !showActual);

  const gridControls = document.querySelector('.grid-controls');
  if (gridControls) {
    gridControls.classList.toggle('hidden', !showActual);
  }
  if (!showActual) {
    state.sortMode = 'predicted';
    const btnPredicted = el('sortByPredicted');
    const btnActual = el('sortByActual');
    if (btnPredicted) btnPredicted.classList.add('active');
    if (btnActual) btnActual.classList.remove('active');
  }

  const sorted = sortDrivers(race.drivers, showActual);
  const body = el('gridBody');
  body.innerHTML = '';

  sorted.forEach((driver, i) => {
    const tr = document.createElement('tr');
    tr.className = 'grid-row';
    tr.style.animationDelay = `${i * 18}ms`;

    const posClass = driver.predicted_pos === 1 ? 'p1' : driver.predicted_pos === 2 ? 'p2' : driver.predicted_pos === 3 ? 'p3' : '';
    const winPct = Math.round(driver.win_prob * 100);
    const dnfPct = Math.round(driver.dnf_risk * 100);
    const dnfClass = dnfPct >= 12 ? 'high' : dnfPct >= 7 ? 'mid' : 'low';
    const winBarWidth = Math.min(winPct * 2.5, 100); // scale so 40% = full bar
    const winHighClass = winPct >= 20 ? 'high' : '';

    let actualHTML = '';
    let accHTML = '';
    if (showActual && driver.actual_pos !== null) {
      const actClass = driver.actual_pos === 1 ? 'p1' : driver.actual_pos === 2 ? 'p2' : driver.actual_pos === 3 ? 'p3' : '';
      actualHTML = `<td><span class="actual-badge ${actClass}">P${driver.actual_pos}</span></td>`;
      const accClass = accClass_(driver.error);
      accHTML = `<td><span class="acc-dot ${accClass}" title="${accLabel_(driver.error)}"></span></td>`;
    } else if (showActual) {
      actualHTML = `<td>—</td>`;
      accHTML = `<td></td>`;
    }

    tr.innerHTML = `
      <td><span class="pos-badge ${posClass}">P${driver.predicted_pos}</span></td>
      <td>
        <div class="driver-cell">
          <div class="driver-team-bar" style="background:${driver.team_color}"></div>
          <div class="driver-info">
            <span class="driver-code">${driver.code}</span>
            <span class="driver-team">${teamName(driver.constructor_id)}</span>
          </div>
        </div>
      </td>
      <td class="win-bar-cell">
        <div class="win-bar-wrap">
          <div class="win-bar-track">
            <div class="win-bar-fill" style="width:${winBarWidth}%"></div>
          </div>
          <span class="win-bar-pct ${winHighClass}">${winPct}%</span>
        </div>
      </td>
      <td class="dnf-cell ${dnfClass}">${dnfPct}%</td>
      ${actualHTML}
      ${accHTML}
    `;
    body.appendChild(tr);
  });
}

function sortDrivers(drivers, showActual) {
  if (state.sortMode === 'actual' && showActual) {
    return [...drivers].sort((a, b) => {
      if (a.actual_pos === null) return 1;
      if (b.actual_pos === null) return -1;
      return a.actual_pos - b.actual_pos;
    });
  }
  return [...drivers].sort((a, b) => a.predicted_pos - b.predicted_pos);
}

// ── TRACK RECORD ─────────────────────────────────────────────
function renderTrackRecord() {
  const agg = state.data.aggregate;
  const winsTotal = agg.races_with_actuals;
  const winsCorrect = agg.winners_correct;
  const winPct = winsTotal > 0 ? Math.round((winsCorrect / winsTotal) * 100) : 0;
  const podiumPct = agg.podium_total > 0 ? Math.round((agg.podium_hits / agg.podium_total) * 100) : 0;
  const top10Pct = agg.top10_total > 0 ? Math.round((agg.top10_hits / agg.top10_total) * 100) : 0;

  el('trackRecord').innerHTML = `
    <div class="stat-pill">
      <span class="stat-pill-label">Winners Called</span>
      <span class="stat-pill-value ${winPct >= 60 ? 'good' : 'ok'}">${winsCorrect}/${winsTotal}</span>
      <span class="stat-pill-sub">${winPct}% correct</span>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-label">Podium Picks</span>
      <span class="stat-pill-value ${podiumPct >= 55 ? 'good' : 'ok'}">${agg.podium_hits}/${agg.podium_total}</span>
      <span class="stat-pill-sub">${podiumPct}% in top 3</span>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-label">Top 10 Picks</span>
      <span class="stat-pill-value good">${agg.top10_hits}/${agg.top10_total}</span>
      <span class="stat-pill-sub">${top10Pct}% in points</span>
    </div>
    <div class="stat-pill">
      <span class="stat-pill-label">Exact Position</span>
      <span class="stat-pill-value">${agg.exact_total}/${agg.n_total}</span>
      <span class="stat-pill-sub">across all races</span>
    </div>
  `;

  // Winner streak
  const streakEl = el('winnerStreak');
  const rounds = state.data.races.filter(r => r.has_actuals);
  streakEl.innerHTML = `
    <div class="streak-label">Winner streak</div>
    <div class="streak-dots">
      ${rounds.map((r, i) => {
        const hit = r.metrics?.winner_correct;
        return `
          <div class="streak-dot">
            <div class="streak-circle ${hit ? 'win' : 'loss'}">${hit ? 'W' : 'L'}</div>
            <span class="streak-round-label">R${r.round}</span>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

// ── TREND CHART ──────────────────────────────────────────────
function renderTrendChart() {
  const races = state.data.races.filter(r => r.has_actuals && r.metrics);
  const labels = races.map(r => `R${r.round}`);
  const podiumHitPct = races.map(r => Math.round((r.metrics.podium_hits / 3) * 100));
  const top10HitPct  = races.map(r => Math.round((r.metrics.top10_hits / 10) * 100));

  const canvas = el('trendChart');
  if (state.charts.trend) { state.charts.trend.destroy(); }

  state.charts.trend = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Podium accuracy %',
          data: podiumHitPct,
          borderColor: '#FFD700',
          backgroundColor: 'rgba(255,215,0,0.08)',
          fill: true,
          tension: 0.4,
          pointRadius: 5,
          pointBackgroundColor: '#FFD700',
        },
        {
          label: 'Top 10 accuracy %',
          data: top10HitPct,
          borderColor: '#00D27A',
          backgroundColor: 'rgba(0,210,122,0.06)',
          fill: true,
          tension: 0.4,
          pointRadius: 5,
          pointBackgroundColor: '#00D27A',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: '#A8A8B8',
            font: { family: 'Titillium Web', size: 12, weight: '600' },
          },
        },
        tooltip: {
          backgroundColor: '#1E1E2A',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#fff',
          bodyColor: '#A8A8B8',
          titleFont: { family: 'Titillium Web', weight: '700' },
          bodyFont: { family: 'Titillium Web', size: 12 },
          callbacks: {
            label: ctx => ` ${ctx.dataset.label}: ${ctx.raw}%`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#606075', font: { family: 'Titillium Web', size: 11 } },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          min: 0, max: 100,
          ticks: {
            color: '#606075',
            font: { family: 'Titillium Web', size: 11 },
            callback: v => `${v}%`,
          },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
      },
    },
  });
}

// ── SHARE PREVIEW ────────────────────────────────────────────
function renderSharePreview(race) {
  const topThree = [...race.drivers]
    .sort((a, b) => a.predicted_pos - b.predicted_pos)
    .slice(0, 3);

  const posLabels = [
    { label: 'P1', cls: 'gold' },
    { label: 'P2', cls: 'silver' },
    { label: 'P3', cls: 'bronze' },
  ];

  el('sharePreview').innerHTML = `
    <div class="share-card-header">
      <span class="share-card-title">F1 PREDICT · 2026</span>
      <span class="share-card-race">${raceCode(race.circuit_id)} — ${race.race_name}</span>
    </div>
    <div class="share-podium-list">
      ${topThree.map((d, i) => `
        <div class="share-podium-row">
          <span class="share-pos ${posLabels[i].cls}">${posLabels[i].label}</span>
          <span class="driver-team-bar" style="background:${d.team_color};width:3px;height:20px;display:inline-block;margin:0 4px"></span>
          <span class="share-driver">${d.code} — ${d.full_name}</span>
          <span class="share-prob">${Math.round(d.win_prob * 100)}% win chance</span>
        </div>
      `).join('')}
    </div>
    <div class="share-footer-text">f1predict.app · Telemetry-driven predictions for analysis</div>
  `;
}

// ════════════════════════════════════════════════════════════════
// SHARE ACTIONS
// ════════════════════════════════════════════════════════════════
window.app = window.app || {};

app.shareCopyText = function() {
  const race = currentRace();
  if (!race) return;
  const topThree = [...race.drivers]
    .sort((a, b) => a.predicted_pos - b.predicted_pos)
    .slice(0, 3);

  const agg = state.data?.aggregate;
  const winsTotal = agg ? agg.races_with_actuals : 0;
  const winsCorrect = agg ? agg.winners_correct : 0;
  const statLine = winsTotal > 0
    ? `${winsCorrect}/${winsTotal} winners predicted correctly in 2026`
    : `2026 Season predictions`;

  const text = [
    `F1 Predict — ${race.race_name}`,
    ``,
    `P1: ${topThree[0]?.code} (${Math.round(topThree[0]?.win_prob * 100)}% win chance)`,
    `P2: ${topThree[1]?.code} (${Math.round(topThree[1]?.win_prob * 100)}%)`,
    `P3: ${topThree[2]?.code} (${Math.round(topThree[2]?.win_prob * 100)}%)`,
    ``,
    `Telemetry-driven F1 predictions | ${statLine}`,
  ].join('\n');

  navigator.clipboard.writeText(text).then(() => showToast());
};

app.shareGenerateImage = function() {
  const preview = el('sharePreview');
  if (typeof html2canvas === 'undefined') {
    app.shareCopyText();
    return;
  }
  html2canvas(preview, {
    backgroundColor: '#000000',
    scale: 2,
  }).then(canvas => {
    canvas.toBlob(blob => {
      if (blob && navigator.clipboard?.write) {
        navigator.clipboard.write([
          new ClipboardItem({ 'image/png': blob }),
        ]).then(showToast).catch(() => {
          // Fallback: download
          const a = document.createElement('a');
          a.href = canvas.toDataURL('image/png');
          a.download = 'f1-predict-picks.png';
          a.click();
          showToast();
        });
      } else {
        // Just download
        const a = document.createElement('a');
        a.href = canvas.toDataURL('image/png');
        a.download = 'f1-predict-picks.png';
        a.click();
        showToast();
      }
    });
  });
};

function showToast() {
  const toast = el('shareToast');
  toast.classList.remove('hidden');
  setTimeout(() => toast.classList.add('hidden'), 2500);
}

// ════════════════════════════════════════════════════════════════
// MODE TOGGLE
// ════════════════════════════════════════════════════════════════
el('btnPreRace').addEventListener('click', () => setMode('pre'));
el('btnPostRace').addEventListener('click', () => setMode('post'));

function setMode(mode) {
  state.mode = mode;
  el('btnPreRace').classList.toggle('active', mode === 'pre');
  el('btnPostRace').classList.toggle('active', mode === 'post');
  const race = currentRace();
  if (race) renderRace(race);
}

// ════════════════════════════════════════════════════════════════
// SORT MODE
// ════════════════════════════════════════════════════════════════
app.setSortMode = function(mode) {
  state.sortMode = mode;
  el('sortByPredicted').classList.toggle('active', mode === 'predicted');
  el('sortByActual').classList.toggle('active', mode === 'actual');
  const race = currentRace();
  if (race) renderGrid(race);
};

// ════════════════════════════════════════════════════════════════
// NERD ZONE
// ════════════════════════════════════════════════════════════════
let nerdBuilt = false;

app.toggleNerdZone = function() {
  state.nerdOpen = !state.nerdOpen;
  el('nerdContent').classList.toggle('hidden', !state.nerdOpen);
  el('nerdArrow').classList.toggle('open', state.nerdOpen);
  el('nerdToggle').setAttribute('aria-expanded', state.nerdOpen);
  
  // Sync sidebar button if present
  const sidebarBtn = el('sidebarNerdToggle');
  if (sidebarBtn) {
    sidebarBtn.setAttribute('aria-expanded', state.nerdOpen);
    const arrow = el('sidebarNerdArrow');
    if (arrow) arrow.classList.toggle('open', state.nerdOpen);
  }
  
  if (state.nerdOpen && !nerdBuilt) {
    buildNerdContent();
    nerdBuilt = true;
  }

  // Smooth scroll to nerd zone section
  if (state.nerdOpen) {
    setTimeout(() => {
      el('nerdZoneSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 80);
  }
};

app.showNerdTab = function(tabName, buttonEl) {
  state.nerdTab = tabName;
  
  // Toggle visibility of all tab panes
  const tabs = ['pipeline', 'features', 'validation', 'tech'];
  tabs.forEach(t => {
    const pane = el(`nerdTab-${t}`);
    if (pane) {
      if (t === tabName) {
        pane.classList.remove('hidden');
      } else {
        pane.classList.add('hidden');
      }
    }
  });
  
  // Update active tab buttons
  document.querySelectorAll('.nerd-tab-btn').forEach(btn => btn.classList.remove('active'));
  if (buttonEl) {
    buttonEl.classList.add('active');
  } else {
    const activeBtn = el(`nerdTabBtn-${tabName}`);
    if (activeBtn) activeBtn.classList.add('active');
  }
  
  // Re-render/resize feature chart if showing features tab so it calculates width correctly
  if (tabName === 'features') {
    setTimeout(() => {
      buildFeatChart(state.data.model, state.featTab);
    }, 20);
  }
};

function renderNerdZone() {
  // Only rebuild if already open (e.g. on round change)
  if (state.nerdOpen) {
    buildNerdContent();
  }
}

function buildNerdContent() {
  const model = state.data.model;
  buildModelArch(model);
  buildFormula(model);
  buildFeatChart(model, state.featTab);
  buildFeatTable(model);
  buildCVInfo(model);
  buildSources(model);
  buildLimitations(model);
  buildAccuracyTable();
  
  // Ensure the active tab pane and navigation button state is correct
  app.showNerdTab(state.nerdTab || 'pipeline');
}

function buildModelArch(model) {
  el('modelArch').innerHTML = model.components.map(c => `
    <div class="model-card">
      <div class="model-card-name">${c.name}</div>
      <div class="model-card-type">${c.type}</div>
      <div class="model-card-desc">${c.purpose}</div>
      <div class="model-card-params">
        ${Object.entries(c.params).map(([k, v]) => `<span class="param-chip">${k}=${v}</span>`).join('')}
      </div>
      <div class="model-card-weight">
        ${c.weight_in_blend > 0 ? `Blend weight: <strong>×${c.weight_in_blend}</strong>` : `Risk penalty: <strong>×${Math.abs(c.weight_in_blend)}</strong> (reduces score)`}
      </div>
      <div class="model-card-desc" style="margin-top:8px;font-size:11px;opacity:0.75">${c.training}</div>
    </div>
  `).join('');
}

function buildFormula(model) {
  const parts = model.blend_formula.split(/([×\+\-\(\)])/g);
  el('formulaBox').innerHTML = `
    <div style="margin-bottom:12px;font-size:13px;color:#A8A8B8">Final ranking score for each driver:</div>
    <div style="font-size:16px;line-height:1.8">
      <span class="formula-highlight">blend_score</span>
      = ( <span style="color:#FFD700">0.6</span> × <span class="formula-highlight">reg_score</span>
      + <span style="color:#27F4D2">0.4</span> × <span class="formula-highlight">win_prob</span> )
      × ( 1 − <span style="color:#FF3333">0.6</span> × <span class="formula-highlight">dnf_risk</span> )
    </div>
    <div style="margin-top:16px;font-size:12px;color:#606075;text-align:left">
      <div>• <code>reg_score</code> = (22 − predicted_position) / 21 &nbsp;→ normalised to 0–1, higher=better</div>
      <div>• <code>win_prob</code> = winner classifier output (0–1)</div>
      <div>• <code>dnf_risk</code> = DNF classifier output (0–1); penalises unreliable drivers</div>
      <div style="margin-top:8px">Drivers are then ranked by descending <code>blend_score</code> to produce the final predicted order.</div>
    </div>
  `;
}

app.showFeatTab = function(tab, btn) {
  state.featTab = tab;
  document.querySelectorAll('.feat-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  buildFeatChart(state.data.model, tab);
};

function buildFeatChart(model, tab) {
  const key = tab === 'reg' ? 'regressor' : tab === 'win' ? 'winner_classifier' : 'dnf_classifier';
  const data = model.feature_importance[key].slice(0, 10); // top 10
  const labels = data.map(d => d.feature);
  const values = data.map(d => +(d.importance * 100).toFixed(2));

  const canvas = el('featChart');
  if (state.charts.feat) { state.charts.feat.destroy(); }

  state.charts.feat = new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance (%)',
        data: values,
        backgroundColor: labels.map((_, i) =>
          i === 0 ? 'rgba(255,24,1,0.8)' : i <= 2 ? 'rgba(255,24,1,0.5)' : 'rgba(255,255,255,0.1)'
        ),
        borderColor: 'rgba(255,24,1,0.4)',
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1E1E2A',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#fff',
          bodyColor: '#A8A8B8',
          titleFont: { family: 'Titillium Web', weight: '700' },
          bodyFont: { family: 'Titillium Web', size: 12 },
          callbacks: {
            label: ctx => ` ${ctx.raw.toFixed(2)}% importance`,
            afterLabel: ctx => {
              const feat = ctx.label;
              const desc = state.data.model.feature_descriptions[feat];
              return desc ? `  ${desc.slice(0, 60)}…` : '';
            },
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: '#606075',
            font: { family: 'Titillium Web', size: 11 },
            callback: v => `${v}%`,
          },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        y: {
          ticks: {
            color: '#A8A8B8',
            font: { family: 'Titillium Web', size: 11 },
          },
          grid: { display: false },
        },
      },
    },
  });
}

function buildFeatTable(model) {
  const rows = model.features.map(f => `
    <tr>
      <td><span class="feat-name">${f}</span></td>
      <td class="feat-desc">${model.feature_descriptions[f] || '—'}</td>
    </tr>
  `).join('');

  el('featTable').innerHTML = `
    <table class="feat-table-el">
      <thead>
        <tr>
          <th style="min-width:220px">Feature</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function buildCVInfo(model) {
  const td = model.training_data;
  el('cvInfo').innerHTML = `
    <div class="cv-grid">
      <div class="cv-stat">
        <div class="cv-stat-label">Training Rows</div>
        <div class="cv-stat-value">${td.total_rows.toLocaleString()}</div>
      </div>
      <div class="cv-stat">
        <div class="cv-stat-label">Seasons</div>
        <div class="cv-stat-value">${td.num_seasons}</div>
      </div>
      <div class="cv-stat">
        <div class="cv-stat-label">Circuits</div>
        <div class="cv-stat-value">${td.num_circuits}</div>
      </div>
      <div class="cv-stat">
        <div class="cv-stat-label">Drivers</div>
        <div class="cv-stat-value">${td.num_drivers}</div>
      </div>
    </div>
    <div class="cv-detail">
      <strong>Seasons covered:</strong> ${td.seasons.join(', ')}<br><br>
      <strong>Cross-validation strategy:</strong> ${model.cross_validation}<br><br>
      This means each fold tests on a <em>complete race</em> not seen in training, preventing the model from learning 
      race-specific patterns and ensuring the reported accuracy reflects genuine generalisation to unseen races.
    </div>
  `;
}

function buildSources(model) {
  el('sourcesGrid').innerHTML = model.data_sources.map(s => `
    <div class="source-card">
      <div class="source-name">${s.name}</div>
      <div class="source-url">${s.url}</div>
      <div class="source-usage">${s.usage}</div>
    </div>
  `).join('');
}

// Tech stack is statically built in HTML to support domains categorization

function buildLimitations(model) {
  el('limitations').innerHTML = model.known_limitations.map(l => `
    <div class="limitation-item">
      <span class="limitation-text">${l}</span>
    </div>
  `).join('');
}


// ════════════════════════════════════════════════════════════════
// HELPERS
// ════════════════════════════════════════════════════════════════
function el(id) { return document.getElementById(id); }
function showEl(id) { el(id).classList.remove('hidden'); }
function hideEl(id) { el(id).classList.add('hidden'); }
function currentRace() {
  return state.data?.races.find(r => r.round === state.selectedRound) ?? null;
}

function teamName(constructorId) {
  const map = {
    'mercedes': 'Mercedes', 'ferrari': 'Ferrari', 'red_bull': 'Red Bull',
    'mclaren': 'McLaren', 'aston_martin': 'Aston Martin', 'alpine': 'Alpine',
    'williams': 'Williams', 'rb': 'RB', 'haas': 'Haas', 'sauber': 'Kick Sauber',
    'audi': 'Audi', 'cadillac': 'Cadillac',
  };
  const id = (constructorId || '').toLowerCase();
  for (const [k, v] of Object.entries(map)) {
    if (id.includes(k)) return v;
  }
  return constructorId || 'Unknown';
}

function accClass_(error) {
  if (error === null || error === undefined) return 'acc-miss';
  const e = Math.abs(error);
  if (e === 0) return 'acc-exact';
  if (e <= 1) return 'acc-close';
  if (e <= 3) return 'acc-fair';
  if (e <= 5) return 'acc-rough';
  return 'acc-miss';
}

function accLabel_(error) {
  if (error === null || error === undefined) return 'No data';
  const e = Math.abs(error);
  if (e === 0) return 'Exact!';
  if (e <= 1) return `±${e} — Very close`;
  if (e <= 3) return `±${e} — Close`;
  if (e <= 5) return `±${e} — Bit off`;
  return `±${e} — Missed`;
}

function raceCode(circuitId) {
  const map = {
    'albert_park': 'AUS',
    'shanghai': 'CHN',
    'suzuka': 'JPN',
    'miami': 'MIA',
    'imola': 'EMI',
    'monaco': 'MON',
  };
  return map[circuitId] || (circuitId || '').substring(0, 3).toUpperCase();
}

// ════════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', boot);
