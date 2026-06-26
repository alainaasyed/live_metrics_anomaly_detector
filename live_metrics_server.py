"""
live_metrics_server.py  —  Autointelie Internship 2026
Nega Sri & Alaina Begum

SQLite-based persistence. One file, grows over time.
No downloads — data viewed in dashboard via API endpoints.

Endpoints:
  GET  /metrics          — live psutil reading
  GET  /history          — all past readings from database
  POST /save             — append one reading to database
  POST /daily_summary    — flush session summary
  GET  /compare          — prev vs current session
  GET  /data/raw         — view Raw Data table (paginated)
  GET  /data/daily       — view Daily Summary table
  GET  /data/anomalies   — view Anomaly Log table
  POST /insights         — IsolationForest ML analysis
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import psutil, time, os, sqlite3, json
from datetime import datetime, date
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

DB_FILE = 'metrics_history.db'

# ══════════════════════════════════════════════════════════════
# DATABASE INIT
# ══════════════════════════════════════════════════════════════
def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Raw readings table
    c.execute('''CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        date_str TEXT NOT NULL,
        time_str TEXT NOT NULL,
        session_id TEXT NOT NULL,
        cpu_percent REAL,
        memory_percent REAL,
        disk_percent REAL,
        cpu_anomaly INTEGER,
        memory_anomaly INTEGER,
        disk_anomaly INTEGER,
        cpu_zscore REAL,
        memory_zscore REAL,
        disk_zscore REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Daily summary table
    c.execute('''CREATE TABLE IF NOT EXISTS daily_summary (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_str TEXT NOT NULL,
        session_id TEXT UNIQUE NOT NULL,
        readings_count INTEGER,
        duration_min REAL,
        cpu_avg REAL, cpu_max REAL, cpu_min REAL, cpu_anomalies INTEGER,
        mem_avg REAL, mem_max REAL, mem_min REAL, mem_anomalies INTEGER,
        disk_avg REAL, disk_max REAL, disk_min REAL, disk_anomalies INTEGER,
        total_anomalies INTEGER,
        anomaly_rate REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Anomaly log table
    c.execute('''CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        date_str TEXT NOT NULL,
        time_str TEXT NOT NULL,
        session_id TEXT NOT NULL,
        metric TEXT,
        value REAL,
        zscore REAL,
        severity TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

_session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

init_db()

# ══════════════════════════════════════════════════════════════
# /metrics  —  single live reading
# ══════════════════════════════════════════════════════════════
@app.route('/metrics')
def metrics():
    cpu  = psutil.cpu_percent(interval=0.2)
    mem  = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    return jsonify({
        'current': {
            'cpu_percent':    cpu,
            'memory_percent': mem,
            'disk_percent':   disk,
            'timestamp':      time.time(),
        }
    })

# ══════════════════════════════════════════════════════════════
# /history  —  load all past readings from database on page refresh
# ══════════════════════════════════════════════════════════════
@app.route('/history')
def get_history():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT timestamp, cpu_percent, memory_percent, disk_percent, '
                  'cpu_anomaly, memory_anomaly, disk_anomaly, session_id '
                  'FROM readings ORDER BY timestamp ASC')
        rows = c.fetchall()
        conn.close()
        
        readings = []
        for r in rows:
            readings.append({
                'timestamp':      r[0],
                'cpu_percent':    r[1],
                'memory_percent': r[2],
                'disk_percent':   r[3],
                'cpu_anomaly':    bool(r[4]),
                'mem_anomaly':    bool(r[5]),
                'disk_anomaly':   bool(r[6]),
                'session':        r[7],
            })
        return jsonify({'readings': readings, 'session': _session_id})
    except Exception as e:
        return jsonify({'readings': [], 'error': str(e)})

# ══════════════════════════════════════════════════════════════
# /save  —  append one reading to database
# ══════════════════════════════════════════════════════════════
@app.route('/save', methods=['POST'])
def save_reading():
    body = request.get_json(force=True)
    ts_unix = float(body.get('timestamp', time.time()))
    dt = datetime.fromtimestamp(ts_unix)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT INTO readings 
                 (timestamp, date_str, time_str, session_id,
                  cpu_percent, memory_percent, disk_percent,
                  cpu_anomaly, memory_anomaly, disk_anomaly,
                  cpu_zscore, memory_zscore, disk_zscore)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (ts_unix,
               dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S'),
               body.get('session', _session_id),
               body.get('cpu', 0), body.get('memory', 0), body.get('disk', 0),
               int(body.get('cpu_anomaly', 0)),
               int(body.get('mem_anomaly', 0)),
               int(body.get('disk_anomaly', 0)),
               body.get('cpu_z'), body.get('mem_z'), body.get('disk_z')))
    
    # If any anomaly, also log to anomalies table
    is_any = body.get('cpu_anomaly') or body.get('mem_anomaly') or body.get('disk_anomaly')
    if is_any:
        for metric, val, z, flag in [
            ('CPU', body.get('cpu'), body.get('cpu_z'), body.get('cpu_anomaly')),
            ('Memory', body.get('memory'), body.get('mem_z'), body.get('mem_anomaly')),
            ('Disk', body.get('disk'), body.get('disk_z'), body.get('disk_anomaly')),
        ]:
            if not flag: continue
            sev = 'ERROR' if (z and float(z) > 5) else 'WARNING' if (z and float(z) > 3.5) else 'INFO'
            c.execute('''INSERT INTO anomalies 
                        (timestamp, date_str, time_str, session_id, metric, value, zscore, severity)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                     (ts_unix, dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S'),
                      body.get('session', _session_id), metric, val, z, sev))
    
    conn.commit()
    conn.close()
    return jsonify({'saved': True})

# ══════════════════════════════════════════════════════════════
# /daily_summary  —  flush session aggregate to database
# ══════════════════════════════════════════════════════════════
@app.route('/daily_summary', methods=['POST'])
def daily_summary():
    body = request.get_json(force=True)
    session = body.get('session', _session_id)
    today = date.today().isoformat()
    
    cpu_vals = [float(x) for x in body.get('cpu', [])]
    mem_vals = [float(x) for x in body.get('memory', [])]
    disk_vals = [float(x) for x in body.get('disk', [])]
    
    if not cpu_vals:
        return jsonify({'saved': False})
    
    n = len(cpu_vals)
    total_a = int(body.get('cpu_anomalies', 0)) + int(body.get('mem_anomalies', 0)) + int(body.get('disk_anomalies', 0))
    rate = round(total_a / n * 100, 2) if n else 0
    dur = round(float(body.get('duration_min', 0)), 1)
    
    def stats(arr):
        return round(sum(arr)/len(arr), 2), round(max(arr), 2), round(min(arr), 2)
    
    ca, cx, cn = stats(cpu_vals)
    ma, mx, mn = stats(mem_vals)
    da, dx, dn = stats(disk_vals)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO daily_summary
                 (date_str, session_id, readings_count, duration_min,
                  cpu_avg, cpu_max, cpu_min, cpu_anomalies,
                  mem_avg, mem_max, mem_min, mem_anomalies,
                  disk_avg, disk_max, disk_min, disk_anomalies,
                  total_anomalies, anomaly_rate)
                 VALUES (?, ?, ?, ?,
                         ?, ?, ?, ?,
                         ?, ?, ?, ?,
                         ?, ?, ?, ?,
                         ?, ?)''',
              (today, session, n, dur,
               ca, cx, cn, body.get('cpu_anomalies', 0),
               ma, mx, mn, body.get('mem_anomalies', 0),
               da, dx, dn, body.get('disk_anomalies', 0),
               total_a, rate))
    conn.commit()
    conn.close()
    return jsonify({'saved': True})

# ══════════════════════════════════════════════════════════════
# /data/raw  —  view Raw Data table (paginated)
# Query params: page=1, limit=100
# ══════════════════════════════════════════════════════════════
@app.route('/data/raw')
def view_raw():
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 100))
    offset = (page - 1) * limit
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM readings')
    total = c.fetchone()[0]
    
    c.execute('''SELECT timestamp, date_str, time_str, session_id,
                        cpu_percent, memory_percent, disk_percent,
                        cpu_anomaly, memory_anomaly, disk_anomaly,
                        cpu_zscore, memory_zscore, disk_zscore
                 FROM readings ORDER BY timestamp DESC LIMIT ? OFFSET ?''',
              (limit, offset))
    rows = c.fetchall()
    conn.close()
    
    headers = ['Timestamp', 'Date', 'Time', 'Session',
               'CPU %', 'Memory %', 'Disk %',
               'CPU Anom', 'Mem Anom', 'Disk Anom',
               'CPU Z', 'Mem Z', 'Disk Z']
    
    data = []
    for r in rows:
        data.append({
            'timestamp': datetime.fromtimestamp(r[0]).isoformat(),
            'date': r[1],
            'time': r[2],
            'session': r[3],
            'cpu': round(r[4], 2) if r[4] else 0,
            'memory': round(r[5], 2) if r[5] else 0,
            'disk': round(r[6], 2) if r[6] else 0,
            'cpu_anomaly': bool(r[7]),
            'mem_anomaly': bool(r[8]),
            'disk_anomaly': bool(r[9]),
            'cpu_z': round(r[10], 2) if r[10] else None,
            'mem_z': round(r[11], 2) if r[11] else None,
            'disk_z': round(r[12], 2) if r[12] else None,
        })
    
    return jsonify({
        'headers': headers,
        'data': data,
        'total': total,
        'page': page,
        'limit': limit,
        'pages': (total + limit - 1) // limit,
    })

# ══════════════════════════════════════════════════════════════
# /data/daily  —  view Daily Summary table
# ══════════════════════════════════════════════════════════════
@app.route('/data/daily')
def view_daily():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''SELECT date_str, session_id, readings_count, duration_min,
                        cpu_avg, cpu_max, cpu_min, cpu_anomalies,
                        mem_avg, mem_max, mem_min, mem_anomalies,
                        disk_avg, disk_max, disk_min, disk_anomalies,
                        total_anomalies, anomaly_rate
                 FROM daily_summary ORDER BY date_str DESC, session_id DESC''')
    rows = c.fetchall()
    conn.close()
    
    data = []
    for r in rows:
        data.append({
            'date': r[0],
            'session': r[1],
            'readings': r[2],
            'duration_min': r[3],
            'cpu_avg': round(r[4], 1), 'cpu_max': round(r[5], 1), 'cpu_min': round(r[6], 1), 'cpu_anom': r[7],
            'mem_avg': round(r[8], 1), 'mem_max': round(r[9], 1), 'mem_min': round(r[10], 1), 'mem_anom': r[11],
            'disk_avg': round(r[12], 1), 'disk_max': round(r[13], 1), 'disk_min': round(r[14], 1), 'disk_anom': r[15],
            'total_anom': r[16],
            'rate': round(r[17], 2),
        })
    
    return jsonify({'data': data, 'count': len(data)})

# ══════════════════════════════════════════════════════════════
# /data/anomalies  —  view Anomaly Log table
# ══════════════════════════════════════════════════════════════
@app.route('/data/anomalies')
def view_anomalies():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''SELECT timestamp, date_str, time_str, session_id,
                        metric, value, zscore, severity
                 FROM anomalies ORDER BY timestamp DESC LIMIT 1000''')
    rows = c.fetchall()
    conn.close()
    
    data = []
    for r in rows:
        data.append({
            'timestamp': datetime.fromtimestamp(r[0]).isoformat(),
            'date': r[1],
            'time': r[2],
            'session': r[3],
            'metric': r[4],
            'value': round(r[5], 2) if r[5] else 0,
            'zscore': round(r[6], 2) if r[6] else None,
            'severity': r[7],
        })
    
    return jsonify({'data': data, 'count': len(data)})

# ══════════════════════════════════════════════════════════════
# /compare  —  prev vs current session comparison
# ══════════════════════════════════════════════════════════════
@app.route('/compare')
def compare():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT date_str, session_id, cpu_avg, cpu_max, mem_avg, mem_max, disk_avg, disk_max, '
              'cpu_anomalies, mem_anomalies, disk_anomalies, total_anomalies, anomaly_rate '
              'FROM daily_summary ORDER BY date_str DESC, session_id DESC LIMIT 2')
    rows = c.fetchall()
    conn.close()
    
    if len(rows) < 2:
        return jsonify({
            'report': 'Not enough sessions yet for comparison.\nRun the dashboard for a second session to see trends.',
            'sessions': len(rows)
        })
    
    curr = rows[0]
    prev = rows[1]
    
    lines = [
        f'Session comparison: {prev[0]} vs {curr[0]}\n',
        '── CPU ──',
        f'Avg:  {prev[2]:.1f}% → {curr[2]:.1f}% ({curr[2]-prev[2]:+.1f}%)',
        f'Max:  {prev[3]:.1f}% → {curr[3]:.1f}% ({curr[3]-prev[3]:+.1f}%)',
        f'Anomalies: {int(prev[8])} → {int(curr[8])}',
        '',
        '── Memory ──',
        f'Avg:  {prev[4]:.1f}% → {curr[4]:.1f}% ({curr[4]-prev[4]:+.1f}%)',
        f'Max:  {prev[5]:.1f}% → {curr[5]:.1f}% ({curr[5]-prev[5]:+.1f}%)',
        f'Anomalies: {int(prev[9])} → {int(curr[9])}',
        '',
        '── Disk ──',
        f'Avg:  {prev[6]:.1f}% → {curr[6]:.1f}% ({curr[6]-prev[6]:+.1f}%)',
        f'Max:  {prev[7]:.1f}% → {curr[7]:.1f}% ({curr[7]-prev[7]:+.1f}%)',
        f'Anomalies: {int(prev[10])} → {int(curr[10])}',
        '',
        '── Overall ──',
        f'Total anomalies: {int(prev[11])} → {int(curr[11])}',
        f'Anomaly rate:    {prev[12]:.1f}% → {curr[12]:.1f}%',
    ]
    
    rate_change = curr[12] - prev[12]
    if rate_change < -1:
        lines.append('\n✓ System is healthier — anomaly rate improved.')
    elif rate_change > 1:
        lines.append('\n⚠ More anomalies — system under higher load.')
    else:
        lines.append('\n→ System behaviour is consistent.')
    
    return jsonify({'report': '\n'.join(lines), 'sessions': len(rows)})

# ══════════════════════════════════════════════════════════════
# /insights  —  IsolationForest ML analysis (same as before)
# ══════════════════════════════════════════════════════════════
@app.route('/insights', methods=['POST'])
def insights():
    body = request.get_json(force=True)
    cpu_arr  = np.array(body.get('cpu',    []), dtype=float)
    mem_arr  = np.array(body.get('memory', []), dtype=float)
    disk_arr = np.array(body.get('disk',   []), dtype=float)
    n = min(len(cpu_arr), len(mem_arr), len(disk_arr))

    MIN_READINGS = 10
    if n < MIN_READINGS:
        return jsonify({
            'not_ready': True, 'needed': MIN_READINGS, 'have': n,
            'report': f'Collecting baseline — need {MIN_READINGS} readings, have {n}.'
        }), 202

    cpu_arr, mem_arr, disk_arr = cpu_arr[-n:], mem_arr[-n:], disk_arr[-n:]
    X_raw = np.column_stack([cpu_arr, mem_arr, disk_arr])
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    clf = IsolationForest(n_estimators=200, contamination=0.05, max_samples='auto', random_state=42)
    clf.fit(X)
    all_scores = clf.score_samples(X)
    scores_mean, scores_std = np.mean(all_scores), max(np.std(all_scores), 1e-9)

    RECENT = min(5, n)
    recent_mean_score = np.mean(all_scores[-RECENT:])
    z_recent = (scores_mean - recent_mean_score) / scores_std
    combined_pct = float(np.clip((z_recent / 3.0) * 100, 0, 100))

    def per_metric_score(arr):
        X1 = arr.reshape(-1, 1)
        sc = StandardScaler()
        X1s = sc.fit_transform(X1)
        clf1 = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        clf1.fit(X1s)
        s = clf1.score_samples(X1s)
        sm, ss = np.mean(s), max(np.std(s), 1e-9)
        return float(np.clip(((sm - np.mean(s[-RECENT:])) / ss / 3.0) * 100, 0, 100))

    cpu_pct, mem_pct, disk_pct = per_metric_score(cpu_arr), per_metric_score(mem_arr), per_metric_score(disk_arr)
    recent_labels = clf.predict(X[-RECENT:])
    n_anomalous = int(np.sum(recent_labels == -1))

    report = _build_report(cpu_arr, mem_arr, disk_arr, cpu_pct, mem_pct, disk_pct, combined_pct, n_anomalous, RECENT, n)

    return jsonify({
        'scores': {'cpu': cpu_pct, 'memory': mem_pct, 'disk': disk_pct, 'combined': combined_pct},
        'anomalous_recent': n_anomalous,
        'report': report,
    })

def _build_report(cpu, mem, disk, cpu_pct, mem_pct, disk_pct, combined_pct, n_anomalous, recent_window, n):
    def severity(p): return 'HIGH' if p >= 60 else 'MODERATE' if p >= 30 else 'NORMAL'
    def trend(arr):
        if len(arr) < 6: return 'stable'
        d = float(arr[-1]) - float(arr[-6])
        return 'rising' if d > 3 else 'falling' if d < -3 else 'stable'
    
    lines = [
        f'System health — {n} readings · IsolationForest (3-metric model)\n',
        f'CPU     : {float(cpu[-1]):.1f}%, trend {trend(cpu)}, anomaly {cpu_pct:.0f}% → {severity(cpu_pct)}',
        f'Memory  : {float(mem[-1]):.1f}%, trend {trend(mem)}, anomaly {mem_pct:.0f}% → {severity(mem_pct)}',
        f'Disk    : {float(disk[-1]):.1f}%, trend {trend(disk)}, anomaly {disk_pct:.0f}% → {severity(disk_pct)}',
        f'\nRecent ({recent_window} readings): {n_anomalous} flagged anomalous.',
    ]
    
    high = [m for m, p in [('CPU', cpu_pct), ('Memory', mem_pct), ('Disk', disk_pct)] if p >= 60]
    if severity(combined_pct) == 'HIGH' or high:
        lines.append(f'\n⚠ ANOMALY: {", ".join(high) if high else "Combined"} showing unusual patterns (score {combined_pct:.0f}%).')
    elif severity(combined_pct) == 'MODERATE':
        lines.append(f'\n⚡ MILD DEVIATION: System slightly above baseline (score {combined_pct:.0f}%).')
    else:
        lines.append(f'\n✓ SYSTEM NORMAL: All metrics within expected range (score {combined_pct:.0f}%).')
    
    return '\n'.join(lines)


if __name__ == '__main__':
    init_db()
    print('─' * 60)
    print('  Live Metrics Server — Autointelie Internship 2026')
    print(f'  Session ID: {_session_id}')
    print(f'  Database: {os.path.abspath(DB_FILE)}')
    print('  Endpoints:')
    print('    GET  /metrics          — live reading')
    print('    GET  /history          — load all past data')
    print('    POST /save             — append reading')
    print('    GET  /data/raw         — view raw readings')
    print('    GET  /data/daily       — view daily summaries')
    print('    GET  /data/anomalies   — view anomaly log')
    print('    GET  /compare          — prev vs current')
    print('    POST /insights         — ML analysis')
    print('─' * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
