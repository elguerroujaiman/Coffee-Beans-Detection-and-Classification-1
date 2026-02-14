// Live stats polling
let polling = false;
let pollInterval = null;

function startDetection() {
    fetch('/start')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('status-badge').textContent = '● Running';
                document.getElementById('status-badge').className = 'badge bg-success fs-5';
                document.getElementById('start-btn').disabled = true;
                document.getElementById('stop-btn').disabled = false;

                // Start polling for live stats
                startPolling();
            } else {
                alert(data.message);
            }
        });
}

function stopDetection() {
    fetch('/stop')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('status-badge').textContent = '● Stopped';
                document.getElementById('status-badge').className = 'badge bg-secondary fs-5';
                document.getElementById('start-btn').disabled = false;
                document.getElementById('stop-btn').disabled = true;

                // Stop polling
                stopPolling();

                // Show success message
                setTimeout(() => {
                    alert('Session saved! Check History to view the report.');
                }, 1000);
            } else {
                alert(data.message);
            }
        });
}

function updateLiveStats() {
    fetch('/live_stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('good-count').textContent = data.good_count || 0;
            document.getElementById('bad-count').textContent = data.bad_count || 0;
            document.getElementById('active-count').textContent = data.active_beans || 0;
            document.getElementById('total-count').textContent = data.total || 0;

            // Update status badge if needed
            if (data.status === 'running') {
                document.getElementById('status-badge').textContent = '● Running';
                document.getElementById('status-badge').className = 'badge bg-success fs-5';
            }
        })
        .catch(error => console.error('Error fetching stats:', error));
}

function startPolling() {
    if (!polling) {
        polling = true;
        updateLiveStats(); // Immediate update
        pollInterval = setInterval(updateLiveStats, 1000); // Update every 1s
    }
}

function stopPolling() {
    if (polling) {
        polling = false;
        clearInterval(pollInterval);
    }
}

// Initial stats load on page load
document.addEventListener('DOMContentLoaded', () => {
    updateLiveStats();
});
