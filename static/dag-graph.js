/**
 * Native SVG DAG graph — incremental live updates from /api/dag/graph.
 */
(function (global) {
    'use strict';

    const NODE_W = 152;
    const NODE_H = 54;
    const STATUS_STYLE = {
        pending: { fill: '#e4e4e7', stroke: '#a1a1aa', sw: 1.5 },
        running: { fill: '#fef3c7', stroke: '#f59e0b', sw: 2.5 },
        complete: { fill: '#dcfce7', stroke: '#22c55e', sw: 1.5 },
        failed: { fill: '#fee2e2', stroke: '#ef4444', sw: 2 },
        skipped: { fill: '#fafafa', stroke: '#d4d4d8', sw: 1 },
    };

    function statusStyle(status) {
        return STATUS_STYLE[status] || STATUS_STYLE.pending;
    }

    function escapeXml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function nodeCenter(n) {
        const p = n.position || { x: 0, y: 0 };
        return { x: p.x + NODE_W / 2, y: p.y + NODE_H / 2 };
    }

    function nodeIds(nodes) {
        return (nodes || []).map(function (n) {
            return n.id;
        });
    }

    function DagGraphController(opts) {
        this.container = opts.container;
        this.onSelect = opts.onSelect || function () {};
        this.onStatus = opts.onStatus || function () {};
        this.view = { scale: 1, tx: 48, ty: 48 };
        this.payload = null;
        this.prevStatus = {};
        this._nodeIds = [];
        this._lastNodeCount = 0;
        this._lastSession = null;
        this._bound = false;
    }

    DagGraphController.prototype.graphUrl = function (sessionId) {
        const base = sessionId
            ? '/api/dag/graph?session_id=' + encodeURIComponent(sessionId)
            : '/api/dag/graph';
        return base + '&_ts=' + Date.now();
    };

    DagGraphController.prototype.destroy = function () {
        if (this.container) this.container.innerHTML = '';
        this.payload = null;
        this.prevStatus = {};
        this._nodeIds = [];
        this._lastNodeCount = 0;
        this._lastSession = null;
        this._didInitialFit = false;
    };

    DagGraphController.prototype.fit = function () {
        if (!this.payload || !this.container) return;
        const nodes = this.payload.nodes || [];
        if (!nodes.length) return;
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        nodes.forEach(function (n) {
            const p = n.position || { x: 0, y: 0 };
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x + NODE_W);
            maxY = Math.max(maxY, p.y + NODE_H);
        });
        const pad = 56;
        const w = this.container.clientWidth || 800;
        const h = this.container.clientHeight || 500;
        const gw = maxX - minX + pad * 2;
        const gh = maxY - minY + pad * 2;
        const scale = Math.min(w / gw, h / gh, 1.4);
        this.view.scale = Math.max(0.15, scale);
        this.view.tx = pad - minX * this.view.scale + (w - gw * this.view.scale) / 2;
        this.view.ty = pad - minY * this.view.scale + (h - gh * this.view.scale) / 2;
        this._applyPanTransform();
    };

    DagGraphController.prototype._applyPanTransform = function () {
        const g = this.container && this.container.querySelector('.dag-graph-pan');
        if (!g) return;
        const v = this.view;
        g.setAttribute('transform', 'translate(' + v.tx + ',' + v.ty + ') scale(' + v.scale + ')');
    };

    DagGraphController.prototype._bindOnce = function () {
        if (this._bound || !this.container) return;
        this._bound = true;
        const self = this;
        let dragging = false;
        let lastX = 0;
        let lastY = 0;

        this.container.addEventListener('wheel', function (e) {
            if (!self.payload) return;
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.08 : 0.92;
            const rect = self.container.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const v = self.view;
            const nx = (mx - v.tx) / v.scale;
            const ny = (my - v.ty) / v.scale;
            v.scale = Math.min(3, Math.max(0.12, v.scale * factor));
            v.tx = mx - nx * v.scale;
            v.ty = my - ny * v.scale;
            self._applyPanTransform();
        }, { passive: false });

        this.container.addEventListener('mousedown', function (e) {
            if (e.button !== 0 || e.target.closest('.dag-svg-node')) return;
            dragging = true;
            lastX = e.clientX;
            lastY = e.clientY;
        });
        global.addEventListener('mousemove', function (e) {
            if (!dragging) return;
            self.view.tx += e.clientX - lastX;
            self.view.ty += e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;
            self._applyPanTransform();
        });
        global.addEventListener('mouseup', function () {
            dragging = false;
        });

        this.container.addEventListener('click', function (e) {
            const nodeEl = e.target.closest('.dag-svg-node');
            if (!nodeEl) return;
            const id = nodeEl.getAttribute('data-dag-id');
            if (id) self.onSelect(id);
        });
    };

    DagGraphController.prototype._liveBadgeHtml = function (payload) {
        var runMode = payload._run_mode || '';
        if (runMode === 'resuming') {
            return '<span class="dag-live-badge dag-live-resume">● RESUMING</span>';
        }
        if (runMode === 'stopped') {
            return '<span class="dag-live-badge dag-live-stopped">■ STOPPED</span>';
        }
        if (runMode === 'new_run' || payload._live) {
            return '<span class="dag-live-badge dag-live-new">● NEW RUN</span>';
        }
        return '';
    };

    DagGraphController.prototype._syncLiveBadge = function (payload) {
        if (!this.container) return;
        var existing = this.container.querySelector('.dag-live-badge');
        var html = this._liveBadgeHtml(payload);
        if (!html) {
            if (existing) existing.remove();
            return;
        }
        if (existing) {
            existing.outerHTML = html;
        } else {
            this.container.insertAdjacentHTML('afterbegin', html);
        }
    };

    DagGraphController.prototype._updateNodesInPlace = function (payload) {
        const nodes = payload.nodes || [];
        const ctrl = this;
        nodes.forEach(function (n) {
            const el = ctrl.container.querySelector('.dag-svg-node[data-dag-id="' + n.id + '"]');
            if (!el) return;
            const st = statusStyle(n.status);
            const rect = el.querySelector('rect');
            if (rect) {
                rect.setAttribute('fill', st.fill);
                rect.setAttribute('stroke', st.stroke);
                rect.setAttribute('stroke-width', String(st.sw));
            }
            const prev = ctrl.prevStatus[n.id];
            const changed = prev !== undefined && prev !== n.status;
            ctrl.prevStatus[n.id] = n.status;
            el.classList.remove('dag-svg-pulse', 'dag-svg-flash');
            if (n.status === 'running') el.classList.add('dag-svg-pulse');
            if (changed) {
                el.classList.add('dag-svg-flash');
                setTimeout(function () {
                    el.classList.remove('dag-svg-flash');
                }, 450);
            }
        });
        this.payload = payload;
        this._syncLiveBadge(payload);
        this.onStatus(
            nodes.length +
                '/' +
                (payload.node_count || nodes.length) +
                ' nodes · ' +
                ((payload.edges && payload.edges.length) || 0) +
                ' edges'
        );
    };

    DagGraphController.prototype._buildEdgeSvg = function (nodes, edges) {
        const nodeById = {};
        nodes.forEach(function (n) {
            nodeById[n.id] = n;
        });
        let edgeSvg = '';
        (edges || []).forEach(function (e) {
            const from = nodeById[e.from];
            const to = nodeById[e.to];
            if (!from || !to) return;
            const a = nodeCenter(from);
            const b = nodeCenter(to);
            const y1 = from.position.y + NODE_H;
            const y2 = to.position.y;
            const mid = (y1 + y2) / 2;
            edgeSvg +=
                '<path class="dag-svg-edge" d="M' +
                a.x +
                ' ' +
                y1 +
                ' C' +
                a.x +
                ' ' +
                mid +
                ', ' +
                b.x +
                ' ' +
                mid +
                ', ' +
                b.x +
                ' ' +
                y2 +
                '" marker-end="url(#dag-arrow)"/>';
        });
        return edgeSvg;
    };

    DagGraphController.prototype._buildNodeSvg = function (nodes) {
        const ctrl = this;
        let nodeSvg = '';
        nodes.forEach(function (n) {
            const st = statusStyle(n.status);
            const p = n.position || { x: 0, y: 0 };
            const lines = String(n.label || n.id).split('\n');
            const title = escapeXml(lines[0] || n.skill);
            const sub = escapeXml(lines[1] || '');
            const prev = ctrl.prevStatus[n.id];
            const changed = prev !== undefined && prev !== n.status;
            ctrl.prevStatus[n.id] = n.status;
            const pulse = n.status === 'running' ? ' dag-svg-pulse' : '';
            const flash = changed ? ' dag-svg-flash' : '';
            nodeSvg +=
                '<g class="dag-svg-node' +
                pulse +
                flash +
                '" data-dag-id="' +
                escapeXml(n.id) +
                '" transform="translate(' +
                p.x +
                ',' +
                p.y +
                ')">' +
                '<rect width="' +
                NODE_W +
                '" height="' +
                NODE_H +
                '" rx="10" fill="' +
                st.fill +
                '" stroke="' +
                st.stroke +
                '" stroke-width="' +
                st.sw +
                '"/>' +
                '<text x="' +
                NODE_W / 2 +
                '" y="20" text-anchor="middle" class="dag-svg-label">' +
                title +
                '</text>' +
                '<text x="' +
                NODE_W / 2 +
                '" y="38" text-anchor="middle" class="dag-svg-sublabel">' +
                sub +
                '</text>' +
                '</g>';
        });
        return nodeSvg;
    };

    DagGraphController.prototype.render = function (payload) {
        if (!this.container) return;
        this._bindOnce();

        const sessionId = payload.session_id || '';
        if (sessionId && sessionId !== this._lastSession) {
            this._lastSession = sessionId;
            this._didInitialFit = false;
            this.prevStatus = {};
            this._nodeIds = [];
            this._lastNodeCount = 0;
        }

        const nodes = payload.nodes || [];
        const edges = payload.edges || [];
        const ids = nodeIds(nodes);
        const sameTopology =
            ids.length === this._nodeIds.length &&
            ids.every(function (id, i) {
                return id === this._nodeIds[i];
            }, this);

        if (!nodes.length) {
            this.container.innerHTML =
                '<p class="dag-graph-empty">Waiting for planner… graph updates automatically during the run.</p>';
            this.payload = payload;
            this._nodeIds = [];
            this._lastNodeCount = 0;
            return;
        }

        if (sameTopology && this.container.querySelector('.dag-graph-svg')) {
            this._updateNodesInPlace(payload);
            return;
        }

        const prevCount = this._lastNodeCount;
        const grew = nodes.length > prevCount;
        this._nodeIds = ids;
        this._lastNodeCount = nodes.length;
        this.payload = payload;

        let minX = 0;
        let minY = 0;
        let maxX = 400;
        let maxY = 400;
        nodes.forEach(function (n) {
            const p = n.position || { x: 0, y: 0 };
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x + NODE_W);
            maxY = Math.max(maxY, p.y + NODE_H);
        });
        const vbPad = 40;
        const viewBox =
            minX -
            vbPad +
            ' ' +
            (minY - vbPad) +
            ' ' +
            (maxX - minX + vbPad * 2) +
            ' ' +
            (maxY - minY + vbPad * 2);

        const edgeSvg = this._buildEdgeSvg(nodes, edges);
        const nodeSvg = this._buildNodeSvg(nodes);
        const liveBadge = this._liveBadgeHtml(payload);

        this.container.innerHTML =
            liveBadge +
            '<svg class="dag-graph-svg" viewBox="' +
            viewBox +
            '" preserveAspectRatio="xMidYMid meet">' +
            '<defs><marker id="dag-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>' +
            '<g class="dag-graph-pan">' +
            '<g class="dag-svg-edges">' +
            edgeSvg +
            '</g><g class="dag-svg-nodes">' +
            nodeSvg +
            '</g></g></svg>';

        this._applyPanTransform();
        const selfRef = this;
        if (!this._didInitialFit || grew) {
            this._didInitialFit = true;
            requestAnimationFrame(function () {
                selfRef.fit();
            });
        }

        this.onStatus(
            nodes.length +
                '/' +
                (payload.node_count || nodes.length) +
                ' nodes · ' +
                edges.length +
                ' edges'
        );
    };

    DagGraphController.prototype.downloadPng = function (filename) {
        const svg = this.container && this.container.querySelector('.dag-graph-svg');
        if (!svg) return false;
        const xml = new XMLSerializer().serializeToString(svg);
        const blob = new Blob([xml], { type: 'image/svg+xml;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = function () {
            const canvas = document.createElement('canvas');
            canvas.width = img.width || 1200;
            canvas.height = img.height || 800;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);
            canvas.toBlob(function (b) {
                if (!b) return;
                const a = document.createElement('a');
                a.href = URL.createObjectURL(b);
                a.download = filename || 'dag-graph.png';
                a.click();
            });
            URL.revokeObjectURL(url);
        };
        img.src = url;
        return true;
    };

    global.DagGraphController = DagGraphController;
})(typeof window !== 'undefined' ? window : global);
