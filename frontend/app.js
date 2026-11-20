/**
 * AgentSwarm Core Client Architecture
 * Connects frontend user flows with FastAPI & PostgreSQL backend.
 */

// Dynamic Backend API Base URL detection
const _EXPLICIT_API_URL = (typeof window.AGENTSWARM_API_URL === 'string')
  ? window.AGENTSWARM_API_URL.trim().replace(/\/+$/, '')
  : '';
const _IS_LOCAL_FRONTEND = !window.location.origin
  || !window.location.origin.startsWith('http')
  || window.location.origin.includes('localhost')
  || window.location.origin.includes('127.0.0.1');
const API_BASE_URL = _EXPLICIT_API_URL
  || (_IS_LOCAL_FRONTEND && !window.location.origin.includes(':8000')
    ? 'http://localhost:8000'
    : (window.location.origin && window.location.origin.startsWith('http')
      ? window.location.origin
      : 'http://localhost:8000'));

/**
 * Robust formatter for API error details, converting arrays/objects to readable strings.
 */
function formatErrorMessage(detail, fallback = 'Operation failed.') {
  if (!detail) return fallback;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map(item => {
      if (typeof item === 'string') return item;
      if (typeof item === 'object' && item !== null) {
        const loc = Array.isArray(item.loc) ? item.loc.filter(l => l !== 'body').join('.') : '';
        const msg = item.msg || item.message || JSON.stringify(item);
        return loc ? `${loc}: ${msg}` : msg;
      }
      return String(item);
    });
    return messages.filter(Boolean).join(' • ') || fallback;
  }
  if (typeof detail === 'object') {
    return detail.msg || detail.message || detail.detail || JSON.stringify(detail);
  }
  return String(detail);
}

/**
 * Safely escape HTML characters before inserting user/API text into templates.
 */
function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Normalize a pasted website URL, prepending https:// when scheme is omitted.
 */
function normalizeTargetUrl(rawUrl, fallback = 'https://staging.store.nexus') {
  let url = (rawUrl || '').trim();
  if (!url) url = fallback;
  if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
  return url;
}

const AgentSwarm = {
  // -----------------------------------------------------------------------
  // 1. User Authentication & Session (Connected to FastAPI & PostgreSQL)
  // -----------------------------------------------------------------------
  auth: {
    currentUser: null,
    token: null,

    /**
     * Authenticate user against PostgreSQL via FastAPI /api/auth/login
     * @param {Object} credentials - { email, password }
     * @returns {Promise<Object>} { success, user, error }
     */
    async handleUserLogin(credentials = {}) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: credentials.email,
            password: credentials.password
          })
        });

        const data = await response.json();

        if (!response.ok) {
          return {
            success: false,
            error: formatErrorMessage(data.detail, 'Login failed. Please check your credentials.')
          };
        }

        // Store JWT token and user info
        this.token = data.access_token;
        this.currentUser = data.user;
        localStorage.setItem('agentswarm_token', data.access_token);
        localStorage.setItem('agentswarm_user', JSON.stringify(data.user));

        this.updateAuthUI();
        return { success: true, user: data.user };
      } catch (err) {
        console.error('[AgentSwarm Auth Error]', err);
        return {
          success: false,
          error: 'Unable to connect to backend server. Ensure FastAPI is running at ' + API_BASE_URL
        };
      }
    },

    /**
     * Register new user account in PostgreSQL via FastAPI /api/auth/signup
     * @param {Object} userData - { email, password, full_name }
     * @returns {Promise<Object>} { success, user, error }
     */
    async handleUserSignup(userData = {}) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            email: userData.email,
            password: userData.password,
            full_name: userData.full_name || null
          })
        });

        const data = await response.json();

        if (!response.ok) {
          return {
            success: false,
            error: formatErrorMessage(data.detail, 'Registration failed. Please check your details.')
          };
        }

        // Auto login on successful registration
        this.token = data.access_token;
        this.currentUser = data.user;
        localStorage.setItem('agentswarm_token', data.access_token);
        localStorage.setItem('agentswarm_user', JSON.stringify(data.user));

        this.updateAuthUI();
        return { success: true, user: data.user };
      } catch (err) {
        console.error('[AgentSwarm Signup Error]', err);
        return {
          success: false,
          error: 'Unable to connect to backend server. Ensure FastAPI is running at ' + API_BASE_URL
        };
      }
    },

    /**
     * Log out current user and clear stored tokens
     */
    handleUserLogout() {
      this.token = null;
      this.currentUser = null;
      localStorage.removeItem('agentswarm_token');
      localStorage.removeItem('agentswarm_user');
      this.updateAuthUI();
    },

    /**
     * Check if user is authenticated from storage and verify with backend
     */
    async checkAuthStatus() {
      const storedToken = localStorage.getItem('agentswarm_token');
      const storedUser = localStorage.getItem('agentswarm_user');

      if (!storedToken) {
        this.updateAuthUI();
        return false;
      }

      this.token = storedToken;
      try {
        this.currentUser = storedUser ? JSON.parse(storedUser) : null;
      } catch (_) {
        this.currentUser = null;
      }
      this.updateAuthUI();

      // Verify token with backend
      try {
        const res = await fetch(`${API_BASE_URL}/api/auth/me`, {
          headers: { 'Authorization': `Bearer ${storedToken}` }
        });
        if (res.ok) {
          const freshUser = await res.json();
          this.currentUser = freshUser;
          localStorage.setItem('agentswarm_user', JSON.stringify(freshUser));
          this.updateAuthUI();
          return true;
        } else {
          // Token expired or invalid
          this.handleUserLogout();
          return false;
        }
      } catch (err) {
        // Network unavailable, retain local session if present
        return !!this.currentUser;
      }
    },

    /**
     * Update navigation header to reflect login state
     */
    updateAuthUI() {
      const container = document.getElementById('auth-nav-container');
      if (!container) return;

      if (this.currentUser) {
        const displayName = this.currentUser.full_name || this.currentUser.email;
        container.innerHTML = `
          <div class="flex items-center gap-2.5">
            <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white border border-border-hairline shadow-subtle text-xs font-mono">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <span class="font-medium text-black truncate max-w-[130px] sm:max-w-[180px]" title="${this.currentUser.email}">
                ${displayName}
              </span>
            </div>
            <button id="btn-signout" type="button" class="text-xs font-mono text-neutral-500 hover:text-black transition-colors px-2 py-1">
              Sign Out
            </button>
            <a class="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-black text-white text-xs font-mono font-medium hover:bg-neutral-800 transition-all whitespace-nowrap shadow-subtle"
              href="workspace.html" id="btn-header-workspace">
              <span>Workspace</span>
              <i class="fa-solid fa-arrow-right text-[9px]"></i>
            </a>
          </div>
        `;
        const signoutBtn = document.getElementById('btn-signout');
        if (signoutBtn) {
          signoutBtn.addEventListener('click', () => this.handleUserLogout());
        }
      } else {
        container.innerHTML = `
          <button
            class="hidden sm:inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg border border-neutral-300 bg-white hover:border-black text-xs font-mono font-semibold text-black shadow-subtle hover:bg-neutral-50 transition-all whitespace-nowrap"
            id="btn-signin" type="button">
            <i class="fa-regular fa-user text-[11px] text-neutral-600"></i>
            <span>Sign In</span>
          </button>
          <a class="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-black text-white text-xs font-mono font-medium hover:bg-neutral-800 transition-all whitespace-nowrap shadow-subtle"
            href="workspace.html" id="btn-header-workspace">
            <span>Launch Workspace</span>
            <i class="fa-solid fa-arrow-right text-[9px]"></i>
          </a>
        `;
        const signInBtn = document.getElementById('btn-signin');
        if (signInBtn) {
          signInBtn.addEventListener('click', () => AgentSwarm.authModal.open('login'));
        }
      }
    }
  },

  // -----------------------------------------------------------------------
  // 2. Auth Modal Controller (Sign In & Sign Up UI)
  // -----------------------------------------------------------------------
  authModal: {
    mode: 'login', // 'login' or 'signup'

    open(mode = 'login') {
      this.mode = mode;
      const modal = document.getElementById('auth-modal-backdrop');
      if (!modal) return;

      this.switchTab(mode);
      this.clearAlert();

      modal.classList.remove('hidden');
      modal.classList.add('flex');
      setTimeout(() => document.getElementById('auth-email')?.focus(), 100);
    },

    close() {
      const modal = document.getElementById('auth-modal-backdrop');
      if (!modal) return;
      modal.classList.add('hidden');
      modal.classList.remove('flex');
      this.clearAlert();
    },

    switchTab(mode) {
      this.mode = mode;
      const tabLogin = document.getElementById('tab-login');
      const tabSignup = document.getElementById('tab-signup');
      const nameWrapper = document.getElementById('field-fullname-wrapper');
      const submitLabel = document.getElementById('auth-submit-label');

      this.clearAlert();

      if (mode === 'login') {
        tabLogin.className = 'py-1.5 rounded-md font-semibold bg-white text-black shadow-subtle transition-all text-center';
        tabSignup.className = 'py-1.5 rounded-md font-medium text-neutral-600 hover:text-black transition-all text-center';
        nameWrapper?.classList.add('hidden');
        if (submitLabel) submitLabel.textContent = 'Sign In to Swarm';
      } else {
        tabSignup.className = 'py-1.5 rounded-md font-semibold bg-white text-black shadow-subtle transition-all text-center';
        tabLogin.className = 'py-1.5 rounded-md font-medium text-neutral-600 hover:text-black transition-all text-center';
        nameWrapper?.classList.remove('hidden');
        if (submitLabel) submitLabel.textContent = 'Create Swarm Account';
      }
    },

    showAlert(message, type = 'error') {
      const alertBox = document.getElementById('auth-alert');
      const alertText = document.getElementById('auth-alert-text');
      if (!alertBox || !alertText) return;

      alertText.textContent = typeof message === 'string' ? message : formatErrorMessage(message);
      alertBox.className = type === 'error'
        ? 'rounded-lg p-3 text-xs font-mono border bg-red-50 text-red-900 border-red-200'
        : 'rounded-lg p-3 text-xs font-mono border bg-emerald-50 text-emerald-900 border-emerald-200';
      alertBox.classList.remove('hidden');
    },

    clearAlert() {
      const alertBox = document.getElementById('auth-alert');
      if (alertBox) alertBox.classList.add('hidden');
    },

    async handleSubmit(event) {
      event.preventDefault();
      const email = document.getElementById('auth-email')?.value.trim();
      const password = document.getElementById('auth-password')?.value;
      const fullName = document.getElementById('auth-fullname')?.value.trim();
      const submitBtn = document.getElementById('btn-auth-submit');
      const submitLabel = document.getElementById('auth-submit-label');

      if (!email || !password) {
        this.showAlert('Please fill in both email and password.');
        return;
      }

      // UI loading state
      if (submitBtn) submitBtn.disabled = true;
      const originalLabel = submitLabel ? submitLabel.textContent : '';
      if (submitLabel) submitLabel.textContent = 'Connecting...';

      let result;
      if (this.mode === 'login') {
        result = await AgentSwarm.auth.handleUserLogin({ email, password });
      } else {
        result = await AgentSwarm.auth.handleUserSignup({ email, password, full_name: fullName });
      }

      if (submitBtn) submitBtn.disabled = false;
      if (submitLabel) submitLabel.textContent = originalLabel;

      if (result.success) {
        this.showAlert(`Welcome ${result.user.email}! Authentication verified.`, 'success');
        setTimeout(() => this.close(), 600);
      } else {
        this.showAlert(result.error || 'Authentication error.');
      }
    }
  },

  // -----------------------------------------------------------------------
  // 3. Cookie Consent & Privacy Compliance (Stubs for future expansion)
  // -----------------------------------------------------------------------
  cookies: {
    initCookieConsent() {
      // Future cookie banner integration
    },
    acceptCookies(preferences = { essential: true, analytics: true }) {},
    declineCookies() {},
    getPreferences() { return null; }
  },

  // -----------------------------------------------------------------------
  // 4. User Data & Profiles
  // -----------------------------------------------------------------------
  userData: {
    async loadUserProfile(userId) { return null; },
    async saveUserSettings(profileData = {}) { return true; }
  },

  // -----------------------------------------------------------------------
  // 4B. AI Model Catalog & Selection Helpers
  // -----------------------------------------------------------------------
  models: {
    catalog: null,

    parseSelection(selectId = 'ws-model-select', customInputId = 'ws-custom-model') {
      const sel = document.getElementById(selectId);
      let raw = (sel?.value || '').trim();
      if (raw === 'custom') {
        const customEl = document.getElementById(customInputId);
        raw = (customEl?.value || '').trim();
      }
      if (!raw) return { provider: null, model_name: null };

      const knownProviders = new Set([
        'google_genai',
        'google',
        'gemini',
        'openrouter',
        'openai',
        'deterministic',
        'local',
        'offline'
      ]);

      if (raw.includes(':')) {
        const idx = raw.indexOf(':');
        const prefix = raw.slice(0, idx).trim().toLowerCase();
        const rest = raw.slice(idx + 1).trim();
        if (knownProviders.has(prefix)) {
          return {
            provider: prefix || null,
            model_name: rest || null
          };
        }
      }

      if (raw.includes('/') || raw.endsWith(':free')) {
        return { provider: 'openrouter', model_name: raw };
      }

      return { provider: null, model_name: raw };
    },

    async loadModels() {
      const wsSelect = document.getElementById('ws-model-select');
      const landingSelect = document.getElementById('landing-model-select');
      if (!wsSelect && !landingSelect) return;

      try {
        const res = await fetch(`${API_BASE_URL}/api/test-runs/models`);
        if (!res.ok) return;
        const data = await res.json();
        this.catalog = data;
        const models = Array.isArray(data.models) ? data.models : [];
        if (!models.length) return;

        const defaultId = `${data.default_provider}:${data.default_model}`;

        if (wsSelect) {
          const prevVal = wsSelect.value;
          const optionsHtml = models.map((m) => {
            const suffix = m.configured ? '' : ' (Add API Key in .env)';
            return `<option value="${escapeHtml(m.id)}">${escapeHtml(m.label + suffix)}</option>`;
          }).join('') + '<option value="custom">Custom Model ID...</option>';
          wsSelect.innerHTML = optionsHtml;
          if (prevVal && [...wsSelect.options].some((o) => o.value === prevVal)) {
            wsSelect.value = prevVal;
          } else if ([...wsSelect.options].some((o) => o.value === defaultId)) {
            wsSelect.value = defaultId;
          }
        }

        if (landingSelect) {
          const prevVal = landingSelect.value;
          landingSelect.innerHTML = models.map((m) => {
            const shortLabel = m.label.replace(' — Fast & Recommended', '').replace(' — Deep Reasoning', '');
            return `<option value="${escapeHtml(m.id)}">${escapeHtml(shortLabel)}</option>`;
          }).join('');
          if (prevVal && [...landingSelect.options].some((o) => o.value === prevVal)) {
            landingSelect.value = prevVal;
          } else if ([...landingSelect.options].some((o) => o.value === defaultId)) {
            landingSelect.value = defaultId;
          }
        }
      } catch (_) {
        // Retain static fallback options if backend is offline
      }
    }
  },

  // -----------------------------------------------------------------------
  // 5. Swarm Execution & Headless Orchestrator API
  // -----------------------------------------------------------------------
  swarm: {
    state: {
      targetUrl: 'https://staging.store.nexus',
      activeAgentIndex: 1, // Default: Workflow Agent (02)
      framework: 'playwright',
      isRunning: false,
      intervalId: null,
      lastRunData: null
    },

    async deploySwarm(rawUrl) {
      const url = normalizeTargetUrl(rawUrl, 'https://staging.store.nexus');
      this.state.targetUrl = url;
      const modelChoice = AgentSwarm.models.parseSelection('landing-model-select');

      const urlInput = document.getElementById('target-url-input');
      if (urlInput) urlInput.value = url;

      const openWsBtn = document.getElementById('btn-open-workspace');
      if (openWsBtn) {
        openWsBtn.href = `workspace.html?url=${encodeURIComponent(url)}`;
      }

      const deployBtn = document.getElementById('btn-deploy-swarm');
      const deployBtnLabel = document.getElementById('btn-deploy-swarm-label');
      const statusText = document.getElementById('cluster-status-text');
      const syncStatus = document.getElementById('sync-status-display');

      if (deployBtn) deployBtn.disabled = true;
      if (deployBtnLabel) deployBtnLabel.textContent = 'Running...';
      if (statusText) statusText.textContent = 'SWARM ACTIVE';
      if (syncStatus) syncStatus.textContent = 'Dispatching LangGraph...';

      let step = 0;
      this.selectAgent(step);

      if (this.state.intervalId) clearInterval(this.state.intervalId);
      this.state.isRunning = true;

      this.state.intervalId = setInterval(() => {
        step = (step + 1) % 5;
        AgentSwarm.swarm.selectAgent(step);
      }, 850);

      try {
        const res = await fetch(`${API_BASE_URL}/api/test-runs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: 'proj-staging-demo',
            target_url: url,
            persona_count: 3,
            use_live_browser: true,
            record_traces: true,
            provider: modelChoice.provider,
            model_name: modelChoice.model_name
          })
        });

        const runData = await res.json();
        if (!res.ok) {
          throw new Error(formatErrorMessage(runData.detail, 'Swarm execution failed.'));
        }

        this.state.lastRunData = runData;

        // Refresh framework spotlight if healing suggestions exist
        const firstPatch = runData?.healing_report?.suggestions?.[0];
        if (firstPatch) {
          ['playwright', 'cypress', 'selenium'].forEach((fw) => {
            if (firstPatch[fw] && frameworkPatches[fw]) {
              frameworkPatches[fw].oldLine = firstPatch.failed_selector || frameworkPatches[fw].oldLine;
              frameworkPatches[fw].newLine = firstPatch[fw].code_snippet || firstPatch[fw].healed_selector;
              frameworkPatches[fw].healedComment = '// ' + (firstPatch[fw].explanation || 'Synthesized by HealingAgent');
            }
          });
        }

        if (openWsBtn) {
          openWsBtn.href = `workspace.html?run_id=${encodeURIComponent(runData.id)}`;
          openWsBtn.innerHTML = '<span>Open Full Workspace Report &rarr;</span>';
        }
        if (syncStatus) {
          syncStatus.textContent = `Run ${runData.id} • ${String(runData.status || 'passed').toUpperCase()}`;
        }
        this.selectAgent(runData.healed_count > 0 ? 3 : 1);
      } catch (err) {
        console.warn('[AgentSwarm Landing Run]', err);
        if (syncStatus) syncStatus.textContent = 'Offline Simulation Mode';
        this.selectAgent(1);
      } finally {
        if (this.state.intervalId) {
          clearInterval(this.state.intervalId);
          this.state.intervalId = null;
        }
        this.state.isRunning = false;
        if (deployBtn) deployBtn.disabled = false;
        if (deployBtnLabel) deployBtnLabel.textContent = 'Deploy Swarm';
        if (statusText) statusText.textContent = 'CLUSTER ONLINE';
      }
    },

    selectAgent(index) {
      this.state.activeAgentIndex = index;
      const agents = [
        {
          id: '01',
          name: 'Explorer Agent',
          badge: 'EXPLORING',
          desc: 'Autonomous crawling of 24 unmapped routes.',
          session: 'Playwright Headless Session #402',
          route: this.state.targetUrl + '/catalog/discover',
          status: '200 OK',
          step: 'Explorer Agent: Crawling Route 18 of 24',
          domState: 'State: Crawling DOM',
          title: 'Dynamic Category Graph (24 Routes Mapped)',
          meta: 'DOM Nodes: 1,420 • Interactive Elements: 86 • Shadow Roots: 3',
          price: '24 Routes',
          env: 'Discovered (Zero Broken Links)',
          cta: 'Traverse Next Subtree',
          action: 'Accessibility Tree Indexed (100%)',
          eventLabel: 'DOM Graph Snapshot',
          conf: 'Coverage: 98.2%',
          failedLabel: '- Unmapped Dynamic Route',
          failedCode: "a[href='/checkout/express-drawer']",
          healedLabel: '+ Autonomous Route Discovery',
          healedCode: 'await agent.explore({ depth: 4, includeShadowDom: true });',
          retest: 'Graph built in 0.64s',
          prStatus: '24 Specs Generated',
          cpu: '42%'
        },
        {
          id: '02',
          name: 'Workflow Agent',
          badge: 'DISPATCHED',
          desc: 'Simulating multi-stage checkout & payment token.',
          session: 'Playwright Headless Session #402',
          route: this.state.targetUrl + '/checkout',
          status: '200 OK',
          step: 'Workflow Agent: Interactive Step 4 of 6',
          domState: 'State: Interactive',
          title: 'Enterprise Precision Laptop (M4 Pro)',
          meta: 'SKU: PROD-9941 • Cart ID: CT-08912',
          price: '$1,499.00',
          env: 'USD (Staging Sandbox)',
          cta: 'Add to Cart & Proceed',
          action: 'Agent Synthetic Click Registered',
          eventLabel: 'Self-Healing Event',
          conf: 'Confidence: 99.4%',
          failedLabel: '- Brittle Legacy Selector (Failed)',
          failedCode: "page.locator('#checkout_btn_v1');",
          healedLabel: '+ Resilient ARIA Healed Locator',
          healedCode: "page.getByRole('button', { name: /proceed/i });",
          retest: 'Re-test: Passed in 0.38s',
          prStatus: 'PR Automated',
          cpu: '34%'
        },
        {
          id: '03',
          name: 'Diagnosis Agent',
          badge: 'ANALYZING',
          desc: 'Triaging GraphQL mutations & DOM anomalies.',
          session: 'Playwright Headless Session #402',
          route: this.state.targetUrl + '/api/graphql',
          status: '200 OK',
          step: 'Diagnosis Agent: Root Cause Triage Step 2 of 2',
          domState: 'State: Inspecting Trace',
          title: 'DOM Refactor vs. API Regression Triage',
          meta: 'GraphQL Mutation: createCheckoutSession • Latency: 48ms',
          price: '0 API Errors',
          env: 'UI Selector Drift Isolated',
          cta: 'Re-verify Network Trace',
          action: 'False-Positive Alert Suppressed',
          eventLabel: 'Root Cause Classification',
          conf: 'Confidence: 99.8%',
          failedLabel: '- Legacy CSS Class Removed',
          failedCode: "Error: Timeout 5000ms waiting for selector '.btn-checkout-legacy'",
          healedLabel: '+ Diagnosis Verdict: Pure Markup Refactor',
          healedCode: 'Backend HTTP 200 verified -> Triggering Self-Healing Agent #04',
          retest: 'Triaged in 0.21s',
          prStatus: 'Zero Flake Noise',
          cpu: '51%'
        },
        {
          id: '04',
          name: 'Healing Agent',
          badge: 'HEALED',
          desc: 'Synthesizing ARIA-resilient locators.',
          session: 'Playwright Headless Session #402',
          route: this.state.targetUrl + '/checkout',
          status: '200 OK',
          step: 'Healing Agent: Semantic ARIA Synthesis Complete',
          domState: 'State: Patch Synthesized',
          title: 'Semantic Accessibility Locator Synthesis',
          meta: 'Target Role: button • Accessible Name: "Add to Cart & Proceed"',
          price: '99.4% Match',
          env: 'Semantic Role Score',
          cta: 'Test Healed Locator',
          action: 'ARIA Locator Bound & Verified',
          eventLabel: 'Self-Healing Event',
          conf: 'Confidence: 99.4%',
          failedLabel: '- Brittle Legacy Selector (Failed)',
          failedCode: "page.locator('#checkout_btn_v1');",
          healedLabel: '+ Resilient ARIA Healed Locator',
          healedCode: "page.getByRole('button', { name: /proceed/i });",
          retest: 'Synthesized in 0.19s',
          prStatus: 'Patch Ready (#402)',
          cpu: '39%'
        },
        {
          id: '05',
          name: 'Verification Agent',
          badge: 'VERIFY',
          desc: 'Regression validation in sandbox runner.',
          session: 'Playwright Headless Session #402',
          route: this.state.targetUrl + '/checkout/summary',
          status: '200 OK',
          step: 'Verification Agent: 10/10 Sandbox Runs Green',
          domState: 'State: Verified Pass',
          title: 'Order Summary & Payment Token Verified',
          meta: 'Sandbox Run: 10/10 Passed • Flake Rate: 0.00%',
          price: '100% Green',
          env: 'Isolated Chromium Cluster',
          cta: 'View Verification Proof',
          action: 'Zero-Hallucination Check Passed',
          eventLabel: 'Sandbox Regression Gate',
          conf: '10/10 Runs Passed',
          failedLabel: '- Before Patch: Suite Status',
          failedCode: 'FAIL e2e/checkout-journey.spec.ts (1 failed, 5000ms timeout)',
          healedLabel: '+ After Patch: Suite Status',
          healedCode: 'PASS e2e/checkout-journey.spec.ts (6 assertions passed in 0.38s)',
          retest: 'Re-test: Passed in 0.38s',
          prStatus: 'PR #402 Ready to Merge',
          cpu: '28%'
        }
      ];

      let active = Object.assign({}, agents[index] || agents[1]);
      const run = this.state.lastRunData;
      if (run) {
        const scenarios = run.test_plan?.scenarios || [];
        const navResults = run.navigation_results || [];
        const diagnoses = run.triage_report?.diagnoses || [];
        const suggestions = run.healing_report?.suggestions || [];
        const totalSteps = navResults.reduce((sum, nr) => sum + (nr.steps?.length || 0), 0);

        active.route = run.target_url || this.state.targetUrl;
        active.session = `Live LangGraph Run • ${run.id}`;

        if (index === 0) {
          active.step = `Planner Agent: ${scenarios.length} Scenarios Generated`;
          active.domState = 'State: Plan Ready';
          active.title = scenarios[0]?.title || run.test_plan?.summary || active.title;
          active.meta = run.test_plan?.summary || `${scenarios.length} scenarios • ${run.personas_count || 0} personas`;
          active.price = `${scenarios.length} Scenarios`;
          active.env = run.target_url;
          active.retest = `Generated for ${run.id}`;
        } else if (index === 1) {
          const firstNav = navResults[0];
          active.step = `Navigator Agent: ${totalSteps} Browser Steps Executed`;
          active.domState = `State: ${String(run.status || 'passed').toUpperCase()}`;
          active.title = firstNav?.scenario_title || active.title;
          active.meta = `Run ${run.id} • Mode: ${firstNav?.execution_mode || 'simulated'} • Traces: ${(run.trace_artifacts || []).length}`;
          active.price = `${totalSteps} Steps`;
          active.env = `Status: ${String(run.status || 'passed').toUpperCase()}`;
          active.retest = `Run status: ${String(run.status || 'passed').toUpperCase()}`;
        } else if (index === 2) {
          const d0 = diagnoses[0];
          active.step = `Diagnosis Agent: ${diagnoses.length} Failure(s) Triaged`;
          active.domState = diagnoses.length ? `Category: ${d0.category}` : 'State: Clean Execution';
          active.title = d0 ? d0.verdict : 'Zero Application Regressions Detected';
          active.meta = d0 ? d0.root_cause_summary : `All ${scenarios.length} planned scenarios verified cleanly.`;
          active.price = `${diagnoses.length} Issues`;
          active.env = d0 ? `Severity: ${d0.severity}` : 'Healthy Run';
          if (d0) {
            active.failedCode = d0.evidence_highlights?.[0] || active.failedCode;
            active.healedCode = d0.root_cause_summary;
          }
        } else if (index === 3) {
          const s0 = suggestions[0];
          active.step = `Healing Agent: ${suggestions.length} Patch(es) Synthesized`;
          active.domState = suggestions.length ? 'State: Healed' : 'State: No Healing Needed';
          if (s0) {
            active.title = 'Resilient ARIA Locator Synthesized';
            active.meta = s0.rationale || active.meta;
            active.price = `${Math.round((s0.confidence || 0.9) * 100)}% Match`;
            active.failedCode = s0.failed_selector;
            active.healedCode = s0.primary_healed_locator;
            active.conf = `Confidence: ${Math.round((s0.confidence || 0.9) * 100)}%`;
          }
        } else if (index === 4) {
          active.step = `Verification Agent: ${(run.trace_artifacts || []).length} Traces Recorded`;
          active.domState = `Final: ${String(run.status || 'passed').toUpperCase()}`;
          active.title = `AgentSwarm Run ${run.id} Complete`;
          active.meta = `Scenarios: ${scenarios.length} • Personas: ${run.personas_count} • Healed: ${run.healed_count}`;
          active.price = String(run.status || 'passed').toUpperCase();
        }
      }

      // Highlight selected agent card
      const cards = document.querySelectorAll('.agent-node');
      cards.forEach((card, idx) => {
        const numEl = card.querySelector('.node-num');
        const badgeEl = card.querySelector('.node-badge');
        if (idx === index) {
          card.className = 'agent-node bg-white p-4 rounded-lg border-2 border-black shadow-subtle transition-all';
          if (idx === 3) {
            numEl.className = 'node-num w-7 h-7 rounded bg-emerald-600 text-white flex items-center justify-center text-xs font-mono font-bold';
            badgeEl.className = 'node-badge text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-600 text-white font-medium';
          } else {
            numEl.className = 'node-num w-7 h-7 rounded bg-black flex items-center justify-center text-white text-xs font-mono';
            badgeEl.className = 'node-badge text-[10px] font-mono px-2 py-0.5 rounded bg-black text-white font-medium';
          }
        } else {
          card.className = 'agent-node bg-white p-4 rounded-lg border border-border-hairline shadow-subtle hover:border-black transition-all';
          if (idx === 3) {
            numEl.className = 'node-num w-7 h-7 rounded bg-emerald-50 text-emerald-800 flex items-center justify-center text-xs font-mono font-bold';
            badgeEl.className = 'node-badge text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 font-medium';
          } else {
            numEl.className = 'node-num w-7 h-7 rounded bg-surface-container flex items-center justify-center text-black text-xs font-mono';
            badgeEl.className = 'node-badge text-[10px] font-mono px-2 py-0.5 rounded bg-surface-container text-neutral-700';
          }
        }
      });

      // Update connection dots
      document.querySelectorAll('.conn-dot').forEach((dot, idx) => {
        if (idx === index) {
          dot.classList.remove('bg-neutral-400');
          dot.classList.add('bg-black', 'scale-125');
        } else {
          dot.classList.remove('bg-black', 'scale-125');
          dot.classList.add('bg-neutral-400');
        }
      });

      // Update inspector values
      const setTxt = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      };

      setTxt('cpu-load-display', 'CPU Load: ' + active.cpu);
      setTxt('inspector-session-title', active.session);
      setTxt('inspector-url-display', active.route);
      setTxt('inspector-http-status', active.status);
      setTxt('inspector-step-label', active.step);
      setTxt('dom-state-badge', active.domState);
      setTxt('dom-product-title', active.title);
      setTxt('dom-product-meta', active.meta);
      setTxt('dom-product-price', active.price);
      setTxt('dom-product-env', active.env);
      setTxt('dom-cta-text', active.cta);
      setTxt('synthetic-action-text', active.action);
      setTxt('healing-event-label', active.eventLabel);
      setTxt('healing-confidence-badge', active.conf);
      setTxt('diff-failed-label', active.failedLabel);
      setTxt('diff-failed-code', active.failedCode);
      setTxt('diff-healed-label', active.healedLabel);
      setTxt('diff-healed-code', active.healedCode);
      setTxt('healing-retest-time', active.retest);
      setTxt('healing-pr-status', active.prStatus);
    }
  },

  // -----------------------------------------------------------------------
  // 6. Dedicated Live Workspace Controller (workspace.html)
  // -----------------------------------------------------------------------
  workspace: {
    state: {
      activeTab: 'plan',
      currentRun: null,
      planPreview: null,
      runsHistory: [],
      projectMap: {},
      lastQuickHeal: null,
      animIntervalId: null,
      activeEventSource: null,
      streamPollId: null
    },

    player: {
      allFrames: [],
      filteredFrames: [],
      currentIndex: 0,
      isPlaying: false,
      playTimerId: null,
      speedMs: 1100,
      scenarioFilter: 'ALL',
      videoUrls: [],
      showRawVideo: false,
      isLive: false,
      seenIndices: new Set(),

      bindControls() {
        document.getElementById('btn-player-play')?.addEventListener('click', () => this.togglePlay());
        document.getElementById('btn-player-prev')?.addEventListener('click', () => {
          this.pause();
          this.goTo(this.currentIndex - 1);
        });
        document.getElementById('btn-player-next')?.addEventListener('click', () => {
          this.pause();
          this.goTo(this.currentIndex + 1);
        });
        document.getElementById('ws-player-scrubber')?.addEventListener('input', (e) => {
          this.pause();
          this.goTo(parseInt(e.target.value, 10) || 0);
        });
        document.getElementById('ws-player-speed')?.addEventListener('change', (e) => {
          this.speedMs = parseInt(e.target.value, 10) || 1100;
          if (this.isPlaying) {
            this.pause();
            this.play();
          }
        });
        document.getElementById('ws-player-scenario-select')?.addEventListener('change', (e) => {
          this.scenarioFilter = e.target.value || 'ALL';
          this.applyFilter(true);
        });
        document.getElementById('btn-player-video-toggle')?.addEventListener('click', () => {
          this.showRawVideo = !this.showRawVideo;
          this.updateVideoModeUI();
        });
      },

      setBadge(mode, text) {
        const badge = document.getElementById('ws-player-live-badge');
        if (!badge) return;
        if (mode === 'live') {
          badge.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-red-50 border border-red-200 text-[10px] font-bold uppercase text-red-700';
          badge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-red-600 animate-ping"></span>${escapeHtml(text)}`;
        } else if (mode === 'ready') {
          badge.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-[10px] font-bold uppercase text-emerald-800';
          badge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-emerald-600"></span>${escapeHtml(text)}`;
        } else {
          badge.className = 'inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-white border border-border-hairline text-[10px] font-bold uppercase text-neutral-700';
          badge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-neutral-400"></span>${escapeHtml(text)}`;
        }
      },

      startLiveStream(targetUrl) {
        this.pause();
        this.allFrames = [];
        this.filteredFrames = [];
        this.currentIndex = 0;
        this.scenarioFilter = 'ALL';
        this.videoUrls = [];
        this.showRawVideo = false;
        this.isLive = true;
        this.seenIndices = new Set();

        this.setBadge('live', 'LIVE STREAMING');
        const urlEl = document.getElementById('ws-player-url');
        if (urlEl) urlEl.textContent = targetUrl || '';
        const httpEl = document.getElementById('ws-player-http');
        if (httpEl) httpEl.textContent = 'CONNECTING';

        document.getElementById('ws-player-empty')?.classList.remove('hidden');
        document.getElementById('ws-player-img')?.classList.add('hidden');
        document.getElementById('ws-player-video')?.classList.add('hidden');
        document.getElementById('ws-player-cursor')?.classList.add('hidden');
        document.getElementById('ws-player-highlight-box')?.classList.add('hidden');
        document.getElementById('ws-player-hud')?.classList.add('hidden');
        document.getElementById('btn-player-video-toggle')?.classList.add('hidden');

        this.populateScenarioSelect();
        this.renderFilmstrip();
        this.updateControlsCounter();
      },

      appendLiveFrame(frame) {
        if (!frame) return;
        const fIdx = frame.frame_index;
        if (fIdx !== undefined && this.seenIndices.has(fIdx)) return;
        if (fIdx !== undefined) this.seenIndices.add(fIdx);

        this.allFrames.push(frame);
        this.populateScenarioSelect();
        this.applyFilter(false);
        // Jump immediately to the latest incoming frame while live streaming
        this.goTo(this.filteredFrames.length - 1);
      },

      loadFromRun(run) {
        this.pause();
        this.isLive = false;
        this.showRawVideo = false;
        this.seenIndices = new Set();

        const frames = [];
        const navResults = run?.navigation_results || [];
        const traceArtifacts = run?.trace_artifacts || [];
        const traceMap = {};
        traceArtifacts.forEach((tr) => {
          if (tr.trace_id) traceMap[tr.trace_id] = tr;
        });

        navResults.forEach((nr) => {
          const steps = nr.steps || [];
          const total = steps.length;
          const trEvents = nr.trace_id && traceMap[nr.trace_id] ? (traceMap[nr.trace_id].events || []) : [];
          steps.forEach((st, idx) => {
            if (st.status === 'skipped') return;
            const trEv = trEvents[idx] || {};
            const shot = st.screenshot_b64 || trEv.screenshot_b64 || null;
            const tBox = st.target_box || trEv.target_box || null;
            frames.push({
              frame_index: frames.length,
              scenario_id: nr.scenario_id,
              scenario_title: nr.scenario_title,
              step_number: st.step_number,
              total_steps: total,
              action_type: st.action?.action_type || 'step',
              selector: st.used_fallback_selector || st.action?.selector || null,
              value: st.action?.value || null,
              description: st.action?.description || st.action?.selector || '',
              status: st.status,
              duration_ms: st.duration_ms || 0,
              current_url: st.current_url || nr.target_url || run.target_url || '',
              error_message: st.error_message || null,
              screenshot_b64: shot,
              target_box: tBox,
              execution_mode: nr.execution_mode || 'simulated'
            });
          });
        });

        this.allFrames = frames;
        this.videoUrls = Array.isArray(run?.video_urls) ? run.video_urls : [];
        this.scenarioFilter = 'ALL';
        this.populateScenarioSelect();
        this.applyFilter(false);
        this.updateVideoModeUI();

        if (this.filteredFrames.length) {
          this.setBadge('ready', `REPLAY READY • ${this.filteredFrames.length} FRAMES`);
          this.goTo(0);
        } else {
          this.setBadge('standby', 'NO FRAMES');
          document.getElementById('ws-player-empty')?.classList.remove('hidden');
          document.getElementById('ws-player-img')?.classList.add('hidden');
          document.getElementById('ws-player-hud')?.classList.add('hidden');
          document.getElementById('ws-player-cursor')?.classList.add('hidden');
          document.getElementById('ws-player-highlight-box')?.classList.add('hidden');
          this.renderFilmstrip();
          this.updateControlsCounter();
        }
      },

      populateScenarioSelect() {
        const sel = document.getElementById('ws-player-scenario-select');
        if (!sel) return;
        const seen = new Map();
        this.allFrames.forEach((f) => {
          if (f.scenario_id && !seen.has(f.scenario_id)) {
            seen.set(f.scenario_id, f.scenario_title || f.scenario_id);
          }
        });
        const prev = this.scenarioFilter;
        let html = '<option value="ALL">All Scenarios</option>';
        seen.forEach((title, id) => {
          html += `<option value="${escapeHtml(id)}">${escapeHtml(id)} — ${escapeHtml(String(title).slice(0, 32))}</option>`;
        });
        sel.innerHTML = html;
        if (prev !== 'ALL' && seen.has(prev)) {
          sel.value = prev;
        } else {
          this.scenarioFilter = 'ALL';
          sel.value = 'ALL';
        }
      },

      applyFilter(resetToStart = true) {
        if (this.scenarioFilter === 'ALL') {
          this.filteredFrames = [...this.allFrames];
        } else {
          this.filteredFrames = this.allFrames.filter((f) => f.scenario_id === this.scenarioFilter);
        }
        this.renderFilmstrip();
        this.updateControlsCounter();
        if (resetToStart && this.filteredFrames.length) {
          this.goTo(0);
        }
      },

      updateVideoModeUI() {
        const toggleBtn = document.getElementById('btn-player-video-toggle');
        const videoEl = document.getElementById('ws-player-video');
        const imgEl = document.getElementById('ws-player-img');
        const cursorEl = document.getElementById('ws-player-cursor');
        const boxEl = document.getElementById('ws-player-highlight-box');
        const hudEl = document.getElementById('ws-player-hud');

        if (toggleBtn) {
          toggleBtn.classList.toggle('hidden', !this.videoUrls.length);
          toggleBtn.innerHTML = this.showRawVideo
            ? '<i class="fa-solid fa-images text-[10px] mr-1"></i>Frame Stream'
            : '<i class="fa-solid fa-video text-[10px] mr-1"></i>Raw .webm';
        }

        if (this.showRawVideo && this.videoUrls.length && videoEl) {
          const fullUrl = this.videoUrls[0].startsWith('http') ? this.videoUrls[0] : `${API_BASE_URL}${this.videoUrls[0]}`;
          if (videoEl.src !== fullUrl) videoEl.src = fullUrl;
          videoEl.classList.remove('hidden');
          imgEl?.classList.add('hidden');
          cursorEl?.classList.add('hidden');
          boxEl?.classList.add('hidden');
          hudEl?.classList.add('hidden');
        } else {
          videoEl?.classList.add('hidden');
          if (this.filteredFrames.length) {
            this.renderFrame(this.currentIndex);
          }
        }
      },

      renderFilmstrip() {
        const strip = document.getElementById('ws-player-filmstrip');
        if (!strip) return;
        if (!this.filteredFrames.length) {
          strip.innerHTML = '<span class="text-neutral-400">Step frames will appear here during execution...</span>';
          return;
        }
        strip.innerHTML = this.filteredFrames.map((f, idx) => {
          const isErr = f.status === 'failed';
          const isActive = idx === this.currentIndex;
          const cls = isActive
            ? 'bg-black text-white border-black'
            : isErr
              ? 'bg-red-50 text-red-800 border-red-200 hover:border-red-500'
              : 'bg-[#faf9f7] text-neutral-700 border-border-hairline hover:border-black';
          return `
            <button type="button" data-frame-idx="${idx}"
              class="ws-filmstrip-pill shrink-0 px-2.5 py-1 rounded border text-[10px] font-mono font-medium transition-all flex items-center gap-1.5 ${cls}">
              <span class="font-bold">#${idx + 1}</span>
              <span class="uppercase">${escapeHtml(f.action_type)}</span>
              <span class="opacity-75 truncate max-w-[110px]">${escapeHtml(f.selector || f.scenario_id || '')}</span>
            </button>
          `;
        }).join('');

        strip.querySelectorAll('.ws-filmstrip-pill').forEach((btn) => {
          btn.addEventListener('click', () => {
            this.pause();
            this.goTo(parseInt(btn.getAttribute('data-frame-idx'), 10) || 0);
          });
        });
      },

      updateControlsCounter() {
        const scrubber = document.getElementById('ws-player-scrubber');
        const counter = document.getElementById('ws-player-counter');
        const total = this.filteredFrames.length;
        if (scrubber) {
          scrubber.max = String(Math.max(0, total - 1));
          scrubber.value = String(Math.min(this.currentIndex, Math.max(0, total - 1)));
        }
        if (counter) {
          counter.textContent = total ? `${this.currentIndex + 1} / ${total}` : '0 / 0';
        }
      },

      goTo(idx) {
        if (!this.filteredFrames.length) return;
        const clamped = Math.max(0, Math.min(idx, this.filteredFrames.length - 1));
        this.currentIndex = clamped;
        this.renderFrame(clamped);
      },

      renderFrame(idx) {
        const frame = this.filteredFrames[idx];
        if (!frame) return;

        const emptyEl = document.getElementById('ws-player-empty');
        const imgEl = document.getElementById('ws-player-img');
        const hudEl = document.getElementById('ws-player-hud');
        const cursorEl = document.getElementById('ws-player-cursor');
        const cursorLbl = document.getElementById('ws-player-cursor-label');
        const boxEl = document.getElementById('ws-player-highlight-box');

        emptyEl?.classList.add('hidden');
        if (!this.showRawVideo) {
          if (imgEl && frame.screenshot_b64) {
            imgEl.src = frame.screenshot_b64;
            imgEl.classList.remove('hidden');
          }
          hudEl?.classList.remove('hidden');
        }

        const setTxt = (id, txt) => {
          const el = document.getElementById(id);
          if (el) el.textContent = txt;
        };
        setTxt('ws-player-url', frame.current_url || '');
        setTxt('ws-player-http', frame.status === 'failed' ? 'FAILED' : 'OK');
        setTxt('ws-player-mode-badge', frame.execution_mode || 'stream');
        setTxt('ws-player-hud-step', `${frame.scenario_id || 'RUN'} • STEP ${frame.step_number || (idx + 1)}/${frame.total_steps || this.filteredFrames.length}`);

        const actionBadge = document.getElementById('ws-player-hud-action');
        if (actionBadge) {
          actionBadge.textContent = String(frame.action_type || 'STEP').toUpperCase();
          actionBadge.className = frame.status === 'failed'
            ? 'px-2 py-0.5 rounded bg-red-600 text-[10px] font-bold uppercase'
            : 'px-2 py-0.5 rounded bg-emerald-600 text-[10px] font-bold uppercase';
        }

        const descTxt = frame.description || frame.selector || frame.value || 'Executing step...';
        setTxt('ws-player-hud-desc', frame.error_message ? `${descTxt} — ${frame.error_message}` : descTxt);
        setTxt('ws-player-hud-duration', `${Number(frame.duration_ms || 0).toFixed(0)} ms`);

        // Animate target highlight box & cursor pointer (no overlapping box on live playwright frames)
        const tBox = frame.target_box;
        const isLivePlaywright = frame.execution_mode === 'playwright';
        if (!this.showRawVideo && tBox && typeof tBox.x_pct === 'number') {
          if (boxEl) {
            if (isLivePlaywright) {
              boxEl.classList.add('hidden');
            } else {
              boxEl.style.left = `${tBox.x_pct}%`;
              boxEl.style.top = `${tBox.y_pct}%`;
              boxEl.style.width = `${tBox.w_pct || 18}%`;
              boxEl.style.height = `${tBox.h_pct || 8}%`;
              boxEl.className = frame.status === 'failed'
                ? 'absolute pointer-events-none rounded-md border-2 border-red-500 bg-red-500/15 shadow-[0_0_0_4px_rgba(220,38,38,0.2)] transition-all duration-300 ease-out z-10'
                : 'absolute pointer-events-none rounded-md border-2 border-emerald-400 bg-emerald-400/15 shadow-[0_0_0_4px_rgba(22,163,74,0.2)] transition-all duration-300 ease-out z-10';
            }
          }
          if (cursorEl) {
            const cx = Math.min(94, tBox.x_pct + (tBox.w_pct || 16) * 0.55);
            const cy = Math.min(90, tBox.y_pct + (tBox.h_pct || 8) * 0.55);
            cursorEl.style.left = `${cx}%`;
            cursorEl.style.top = `${cy}%`;
            cursorEl.classList.remove('hidden');
          }
          if (cursorLbl) {
            const act = String(frame.action_type || 'ACTION').toUpperCase();
            const targetHint = frame.selector || frame.value || '';
            cursorLbl.textContent = targetHint ? `${act} ${String(targetHint).slice(0, 26)}` : act;
          }
        } else {
          boxEl?.classList.add('hidden');
          cursorEl?.classList.add('hidden');
        }

        this.updateControlsCounter();
        this.renderFilmstrip();
      },

      togglePlay() {
        if (this.isPlaying) {
          this.pause();
        } else {
          this.play();
        }
      },

      play() {
        if (!this.filteredFrames.length) return;
        if (this.currentIndex >= this.filteredFrames.length - 1) {
          this.goTo(0);
        }
        this.isPlaying = true;
        const icon = document.getElementById('ws-player-play-icon');
        const label = document.getElementById('ws-player-play-label');
        if (icon) icon.className = 'fa-solid fa-pause text-[10px]';
        if (label) label.textContent = 'Pause';

        if (this.playTimerId) clearInterval(this.playTimerId);
        this.playTimerId = setInterval(() => {
          if (this.currentIndex < this.filteredFrames.length - 1) {
            this.goTo(this.currentIndex + 1);
          } else {
            this.pause();
          }
        }, this.speedMs);
      },

      pause() {
        this.isPlaying = false;
        if (this.playTimerId) {
          clearInterval(this.playTimerId);
          this.playTimerId = null;
        }
        const icon = document.getElementById('ws-player-play-icon');
        const label = document.getElementById('ws-player-play-label');
        if (icon) icon.className = 'fa-solid fa-play text-[10px]';
        if (label) label.textContent = 'Play';
      }
    },

    init() {
      const launchForm = document.getElementById('workspace-launch-form');
      if (!launchForm) return;

      this.checkHealth();
      this.player.bindControls();
      this.bindEvents();
      this.initFromQueryParams();
      this.loadHistory();
    },

    bindEvents() {
      const launchForm = document.getElementById('workspace-launch-form');
      launchForm?.addEventListener('submit', (e) => {
        e.preventDefault();
        this.runSwarmFromWorkspace();
      });

      const modelSelect = document.getElementById('ws-model-select');
      const customModelInput = document.getElementById('ws-custom-model');
      modelSelect?.addEventListener('change', () => {
        if (customModelInput) {
          customModelInput.classList.toggle('hidden', modelSelect.value !== 'custom');
        }
      });

      const planOnlyBtn = document.getElementById('btn-ws-plan-only');
      planOnlyBtn?.addEventListener('click', () => this.generatePlanOnly());

      document.querySelectorAll('.ws-preset-url').forEach((btn) => {
        btn.addEventListener('click', () => {
          const url = btn.getAttribute('data-preset-url') || '';
          const input = document.getElementById('ws-target-url');
          if (input && url) {
            input.value = url;
            input.focus();
          }
        });
      });

      document.querySelectorAll('.ws-tab-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
          const tab = btn.getAttribute('data-ws-tab');
          if (tab) this.switchTab(tab);
        });
      });

      const refreshBtn = document.getElementById('btn-ws-refresh-history');
      refreshBtn?.addEventListener('click', () => this.loadHistory());

      const quickHealForm = document.getElementById('ws-quick-heal-form');
      quickHealForm?.addEventListener('submit', (e) => this.quickHealLocator(e));
    },

    async initFromQueryParams() {
      const params = new URLSearchParams(window.location.search);
      const runId = params.get('run_id');
      const targetUrl = params.get('url');

      if (targetUrl) {
        const input = document.getElementById('ws-target-url');
        if (input) input.value = normalizeTargetUrl(targetUrl);
      }

      if (runId) {
        await this.loadRunById(runId);
      }
    },

    async checkHealth() {
      const dot = document.getElementById('ws-health-dot');
      const text = document.getElementById('ws-health-text');
      if (!dot || !text) return;

      try {
        const res = await fetch(`${API_BASE_URL}/api/health`);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        dot.className = 'w-2 h-2 rounded-full bg-emerald-500';
        const dbLabel = data.database === 'connected' ? 'DB Connected' : 'API Ready';
        text.textContent = `Backend Online • ${dbLabel}`;
      } catch (_) {
        dot.className = 'w-2 h-2 rounded-full bg-red-500';
        text.textContent = 'Backend Unreachable';
      }
    },

    showBanner(message, type = 'info') {
      const el = document.getElementById('ws-status-banner');
      if (!el) return;
      const styles = {
        info: 'rounded-lg px-4 py-3 text-xs font-mono border bg-white text-neutral-800 border-border-hairline',
        success: 'rounded-lg px-4 py-3 text-xs font-mono border bg-emerald-50 text-emerald-900 border-emerald-200',
        error: 'rounded-lg px-4 py-3 text-xs font-mono border bg-red-50 text-red-900 border-red-200'
      };
      el.className = styles[type] || styles.info;
      el.textContent = message;
      el.classList.remove('hidden');
    },

    switchTab(tabName) {
      this.state.activeTab = tabName;
      document.querySelectorAll('.ws-tab-btn').forEach((btn) => {
        const isTarget = btn.getAttribute('data-ws-tab') === tabName;
        btn.className = isTarget
          ? 'ws-tab-btn px-3.5 py-2 rounded-t-lg font-semibold bg-white text-black border border-b-0 border-border-hairline'
          : 'ws-tab-btn px-3.5 py-2 rounded-t-lg font-medium text-neutral-600 hover:text-black';
      });
      document.querySelectorAll('.ws-tab-panel').forEach((panel) => {
        panel.classList.toggle('hidden', panel.id !== `tab-panel-${tabName}`);
      });
    },

    setStageActive(activeIdx, message = null) {
      document.querySelectorAll('.ws-stage-card').forEach((card, idx) => {
        const badge = document.getElementById(`ws-stage-badge-${idx}`);
        const stat = document.getElementById(`ws-stage-stat-${idx}`);
        if (idx === activeIdx) {
          card.className = 'ws-stage-card bg-white p-4 rounded-xl border-2 border-black shadow-subtle transition-all';
          if (badge) {
            badge.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-black text-white font-semibold';
            badge.textContent = 'RUNNING';
          }
          if (stat && message) stat.textContent = message;
        } else {
          card.className = 'ws-stage-card bg-white p-4 rounded-xl border border-border-hairline shadow-subtle transition-all';
          if (badge) {
            badge.className = idx < activeIdx
              ? 'text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 font-medium'
              : 'text-[10px] font-mono px-2 py-0.5 rounded bg-[#f4f3f1] text-neutral-600';
            badge.textContent = idx < activeIdx ? 'DONE' : 'QUEUED';
          }
        }
      });
    },

    startPipelineAnimation() {
      this.setStageActive(0);
    },

    stopStreamListeners() {
      if (this.state.activeEventSource) {
        try {
          this.state.activeEventSource.close();
        } catch (_) {}
        this.state.activeEventSource = null;
      }
      if (this.state.streamPollId) {
        clearInterval(this.state.streamPollId);
        this.state.streamPollId = null;
      }
    },

    seenLogIndices: new Set(),
    partialCache: {},

    clearAgentFeed() {
      this.seenLogIndices = new Set();
      this.partialCache = {};
      const listEl = document.getElementById('ws-agent-feed-list');
      const cntEl = document.getElementById('ws-agent-feed-count');
      if (listEl) {
        listEl.innerHTML = '<div class="text-neutral-500 p-2">Connecting to AgentSwarm live stream...</div>';
      }
      if (cntEl) cntEl.textContent = '0 events';
    },

    appendAgentLog(entry) {
      if (!entry) return;
      const idx = entry.index;
      if (idx !== undefined && this.seenLogIndices.has(idx)) return;
      if (idx !== undefined) this.seenLogIndices.add(idx);

      const listEl = document.getElementById('ws-agent-feed-list');
      const cntEl = document.getElementById('ws-agent-feed-count');
      if (!listEl) return;

      if (this.seenLogIndices.size <= 1) {
        listEl.innerHTML = '';
      }

      const st = entry.status || 'info';
      const badgeCls = st === 'success'
        ? 'bg-emerald-100 text-emerald-900 border-emerald-200'
        : st === 'error'
          ? 'bg-red-100 text-red-900 border-red-200'
          : st === 'warning'
            ? 'bg-amber-100 text-amber-900 border-amber-200'
            : 'bg-white text-neutral-800 border-border-hairline';

      const timeStr = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
      const row = document.createElement('div');
      row.className = 'p-2.5 rounded-lg bg-white border border-border-hairline space-y-1 shadow-subtle';
      row.innerHTML = `
        <div class="flex items-center justify-between gap-2">
          <span class="px-1.5 py-0.5 rounded border text-[9px] font-bold uppercase ${badgeCls}">${escapeHtml(entry.agent || 'Agent')}</span>
          <span class="text-[9px] text-neutral-400">${escapeHtml(timeStr)}</span>
        </div>
        <p class="text-[11px] text-neutral-800 leading-snug break-words">${escapeHtml(entry.message || '')}</p>
      `;
      listEl.appendChild(row);
      listEl.scrollTop = listEl.scrollHeight;
      if (cntEl) cntEl.textContent = `${listEl.children.length} events`;
    },

    handlePartialUpdate(key, val) {
      if (!key || !val) return;
      this.partialCache[key] = val;
      if (key === 'test_plan') {
        const goals = val.goals || val.scenarios || [];
        const scEl = document.getElementById('ws-kpi-scenarios');
        if (scEl) scEl.textContent = String(goals.length);
        this.renderPlanTab(val);
      } else if (key === 'personas') {
        const pList = val.personas || [];
        const pEl = document.getElementById('ws-kpi-personas');
        if (pEl) pEl.textContent = String(pList.length);
        this.renderPersonasTab(val);
      } else if (key === 'navigation_results') {
        const totalSteps = val.reduce((acc, nr) => acc + (nr.steps?.length || 0), 0);
        const stEl = document.getElementById('ws-kpi-steps');
        if (stEl) stEl.textContent = String(totalSteps);
        this.renderTracesTab(val, []);
      } else if (key === 'triage_report' || key === 'healing_report') {
        this.renderHealingTab(this.partialCache.triage_report, this.partialCache.healing_report);
      }
    },

    connectLiveStream(streamId, targetUrl) {
      this.stopStreamListeners();
      this.player.startLiveStream(targetUrl);
      this.clearAgentFeed();

      if (typeof EventSource !== 'undefined') {
        try {
          const es = new EventSource(`${API_BASE_URL}/api/test-runs/stream/${encodeURIComponent(streamId)}`);
          this.state.activeEventSource = es;

          es.addEventListener('stage', (ev) => {
            try {
              const payload = JSON.parse(ev.data);
              if (typeof payload.stage_index === 'number') {
                this.setStageActive(payload.stage_index, payload.message);
              }
            } catch (_) {}
          });

          es.addEventListener('agent_log', (ev) => {
            try {
              const entry = JSON.parse(ev.data);
              this.appendAgentLog(entry);
            } catch (_) {}
          });

          es.addEventListener('partial', (ev) => {
            try {
              const payload = JSON.parse(ev.data);
              this.handlePartialUpdate(payload.key, payload.value);
            } catch (_) {}
          });

          es.addEventListener('frame', (ev) => {
            try {
              const frame = JSON.parse(ev.data);
              this.player.appendLiveFrame(frame);
              const stEl = document.getElementById('ws-kpi-steps');
              if (stEl) stEl.textContent = String(this.player.allFrames.length);
            } catch (_) {}
          });

          es.addEventListener('complete', () => {
            this.stopStreamListeners();
          });

          es.onerror = () => {
            // Polling fallback below keeps frames flowing if SSE closes early
          };
        } catch (_) {}
      }

      this.state.streamPollId = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE_URL}/api/test-runs/stream-state/${encodeURIComponent(streamId)}`);
          if (!res.ok) return;
          const st = await res.json();
          if (typeof st.stage_index === 'number' && st.status === 'running') {
            this.setStageActive(st.stage_index, st.stage_message);
          }
          if (Array.isArray(st.logs)) {
            st.logs.forEach((lg) => this.appendAgentLog(lg));
          }
          if (st.partial_results && typeof st.partial_results === 'object') {
            Object.entries(st.partial_results).forEach(([k, v]) => {
              this.handlePartialUpdate(k, v);
            });
          }
          if (Array.isArray(st.frames)) {
            st.frames.forEach((fr) => this.player.appendLiveFrame(fr));
            const stEl = document.getElementById('ws-kpi-steps');
            if (stEl && st.frames.length) stEl.textContent = String(st.frames.length);
          }
          if (st.status === 'completed' || st.status === 'failed') {
            this.stopStreamListeners();
          }
        } catch (_) {}
      }, 450);
    },

    stopPipelineAnimation(run) {
      if (this.state.animIntervalId) {
        clearInterval(this.state.animIntervalId);
        this.state.animIntervalId = null;
      }
      this.stopStreamListeners();
      const goals = run?.test_plan?.goals || run?.test_plan?.scenarios || [];
      const personas = run?.personas?.personas || [];
      const traces = run?.trace_artifacts || [];
      const diagnoses = run?.triage_report?.diagnoses || [];
      const suggestions = run?.healing_report?.suggestions || [];

      const updates = [
        { badge: 'DONE', stat: `${goals.length} goal(s) set` },
        { badge: 'DONE', stat: `${personas.length} persona(s) created` },
        { badge: 'DONE', stat: `${traces.length} trace artifact(s)` },
        { badge: diagnoses.length ? 'TRIAGED' : 'CLEAN', stat: `${diagnoses.length} failure(s) analyzed` },
        { badge: suggestions.length ? 'HEALED' : 'STANDBY', stat: `${suggestions.length} locator patch(es)` }
      ];

      document.querySelectorAll('.ws-stage-card').forEach((card, idx) => {
        card.className = 'ws-stage-card bg-white p-4 rounded-xl border border-border-hairline shadow-subtle transition-all';
        const badge = document.getElementById(`ws-stage-badge-${idx}`);
        const stat = document.getElementById(`ws-stage-stat-${idx}`);
        if (badge && updates[idx]) {
          badge.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 font-semibold';
          badge.textContent = updates[idx].badge;
        }
        if (stat && updates[idx]) {
          stat.textContent = updates[idx].stat;
        }
      });
    },

    parseWorkspaceInputs() {
      const rawUrl = document.getElementById('ws-target-url')?.value || '';
      const targetUrl = normalizeTargetUrl(rawUrl, 'https://example.com');
      const urlEl = document.getElementById('ws-target-url');
      if (urlEl) urlEl.value = targetUrl;

      const rawObj = document.getElementById('ws-objectives')?.value || '';
      const objectives = rawObj.split(',').map((s) => s.trim()).filter(Boolean);
      const personaCount = parseInt(document.getElementById('ws-persona-count')?.value || '3', 10);
      const useLiveBrowser = Boolean(document.getElementById('ws-live-browser')?.checked);
      const brokenSelector = (document.getElementById('ws-broken-selector')?.value || '').trim();
      const modelChoice = AgentSwarm.models.parseSelection('ws-model-select', 'ws-custom-model');

      const rawFailures = [];
      if (brokenSelector) {
        rawFailures.push({
          failure_id: 'SIM-BRK-001',
          scenario_id: 'SCN-SIM',
          scenario_title: 'Simulated Broken Selector Verification',
          step_number: 1,
          target_url: targetUrl,
          error_message: `Timeout 5000ms waiting for selector '${brokenSelector}'`,
          failed_selector: brokenSelector,
          http_status: 200,
          console_errors: [],
          network_failures: [],
          action_description: `Click target control (${brokenSelector})`
        });
      }

      return {
        targetUrl,
        objectives,
        personaCount,
        useLiveBrowser,
        rawFailures,
        provider: modelChoice.provider,
        modelName: modelChoice.model_name
      };
    },

    async runSwarmFromWorkspace() {
      const { targetUrl, objectives, personaCount, useLiveBrowser, rawFailures, provider, modelName } = this.parseWorkspaceInputs();
      const runBtn = document.getElementById('btn-ws-run');
      const runLabel = document.getElementById('btn-ws-run-label');
      const streamId = 'stm-' + Math.random().toString(36).slice(2, 10);

      if (runBtn) runBtn.disabled = true;
      if (runLabel) runLabel.textContent = 'Streaming Swarm...';
      this.showBanner(`Streaming 5-agent LangGraph swarm against ${targetUrl} (${modelName || 'default model'})...`, 'info');
      this.startPipelineAnimation();
      this.connectLiveStream(streamId, targetUrl);

      try {
        const res = await fetch(`${API_BASE_URL}/api/test-runs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            project_id: 'proj-staging-demo',
            target_url: targetUrl,
            objectives,
            persona_count: personaCount,
            raw_failures: rawFailures,
            use_live_browser: useLiveBrowser,
            record_traces: true,
            provider,
            model_name: modelName,
            stream_id: streamId
          })
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(formatErrorMessage(data.detail, 'Swarm run failed.'));
        }

        this.state.currentRun = data;
        this.stopPipelineAnimation(data);
        this.renderRunResult(data);
        this.showBanner(
          `Completed run ${data.id} on ${data.target_url} • Model: ${data.model_name || 'auto'} • Status: ${String(data.status).toUpperCase()} • ${data.scenarios_count} scenarios, ${data.failures_count} failures, ${data.healed_count} healed.`,
          data.status === 'failed' ? 'error' : 'success'
        );
        await this.loadHistory();
      } catch (err) {
        this.stopPipelineAnimation(this.state.currentRun || {});
        this.showBanner(err.message || 'Unable to execute AgentSwarm run.', 'error');
      } finally {
        if (runBtn) runBtn.disabled = false;
        if (runLabel) runLabel.textContent = 'Run AgentSwarm on URL';
      }
    },

    async generatePlanOnly() {
      const { targetUrl, objectives, personaCount, useLiveBrowser, provider, modelName } = this.parseWorkspaceInputs();
      const planBtn = document.getElementById('btn-ws-plan-only');
      const planLabel = document.getElementById('btn-ws-plan-label');

      if (planBtn) planBtn.disabled = true;
      if (planLabel) planLabel.textContent = 'Planning...';
      this.showBanner(`Generating QA test plan and personas for ${targetUrl}...`, 'info');

      try {
        let projectId = 'proj-staging-demo';
        if (targetUrl !== 'https://staging.store.nexus') {
          if (this.state.projectMap[targetUrl]) {
            projectId = this.state.projectMap[targetUrl];
          } else {
            let hostName = targetUrl;
            try {
              hostName = new URL(targetUrl).hostname;
            } catch (_) {}
            const projRes = await fetch(`${API_BASE_URL}/api/projects`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                name: hostName.slice(0, 80),
                target_url: targetUrl,
                description: `Workspace target ${targetUrl}`
              })
            });
            const projData = await projRes.json();
            if (!projRes.ok) {
              throw new Error(formatErrorMessage(projData.detail, 'Failed creating workspace project.'));
            }
            projectId = projData.id;
            this.state.projectMap[targetUrl] = projectId;
          }
        }

        const res = await fetch(`${API_BASE_URL}/api/projects/${encodeURIComponent(projectId)}/generate-plan`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            objectives: objectives.length ? objectives : null,
            persona_count: personaCount,
            provider,
            model_name: modelName,
            use_live_browser: useLiveBrowser
          })
        });
        const preview = await res.json();
        if (!res.ok) {
          throw new Error(formatErrorMessage(preview.detail, 'Failed generating test plan.'));
        }

        const syntheticRun = {
          id: 'plan-preview',
          project_id: preview.project_id,
          target_url: preview.target_url,
          provider: preview.provider,
          model_name: preview.model_name,
          status: 'passed',
          scenarios_count: (preview.test_plan?.scenarios || []).length,
          personas_count: (preview.personas?.personas || []).length,
          failures_count: 0,
          healed_count: 0,
          test_plan: preview.test_plan,
          personas: preview.personas,
          navigation_results: [],
          trace_artifacts: [],
          triage_report: { diagnoses: [] },
          healing_report: null
        };
        this.state.planPreview = preview;
        this.renderRunResult(syntheticRun, true);
        this.stopPipelineAnimation(syntheticRun);
        this.switchTab('plan');
        this.showBanner(`Generated ${syntheticRun.scenarios_count} scenarios & ${syntheticRun.personas_count} personas for ${preview.target_url}.`, 'success');
      } catch (err) {
        this.showBanner(err.message || 'Plan generation failed.', 'error');
      } finally {
        if (planBtn) planBtn.disabled = false;
        if (planLabel) planLabel.textContent = 'Generate Test Plan Only';
      }
    },

    renderRunResult(run, isPlanPreview = false) {
      const kpiStatus = document.getElementById('ws-kpi-status');
      const statusStr = isPlanPreview ? 'PLAN PREVIEW' : String(run.status || 'passed').toUpperCase();
      if (kpiStatus) {
        const badgeClass = run.status === 'failed'
          ? 'inline-flex items-center px-2.5 py-0.5 rounded text-xs font-mono font-bold bg-red-100 text-red-900'
          : run.status === 'healed'
            ? 'inline-flex items-center px-2.5 py-0.5 rounded text-xs font-mono font-bold bg-emerald-100 text-emerald-900'
            : 'inline-flex items-center px-2.5 py-0.5 rounded text-xs font-mono font-bold bg-emerald-50 text-emerald-800';
        kpiStatus.className = badgeClass;
        kpiStatus.textContent = statusStr;
      }

      const goals = run.test_plan?.goals || run.test_plan?.scenarios || [];
      const personas = run.personas?.personas || [];
      const navResults = run.navigation_results || [];
      const totalSteps = navResults.reduce((acc, nr) => acc + (nr.steps?.length || 0), 0);

      const setTxt = (id, txt) => {
        const el = document.getElementById(id);
        if (el) el.textContent = txt;
      };
      setTxt('ws-kpi-scenarios', String(run.scenarios_count ?? goals.length));
      setTxt('ws-kpi-personas', String(run.personas_count ?? personas.length));
      setTxt('ws-kpi-steps', String(totalSteps));
      setTxt('ws-kpi-healed', `${run.healed_count ?? 0} / ${run.failures_count ?? 0}`);
      setTxt('ws-kpi-run-id', isPlanPreview ? 'Preview Only (not saved to run history)' : `Run ID: ${run.id}`);
      const modelLabel = run.model_name ? `${run.provider || 'llm'}:${run.model_name}` : 'Auto';
      setTxt('ws-kpi-model', `Model: ${modelLabel}`);

      const kpiUrl = document.getElementById('ws-kpi-url');
      if (kpiUrl) {
        kpiUrl.textContent = run.target_url || '';
        kpiUrl.href = run.target_url || '#';
      }

      if (!isPlanPreview) {
        this.player.loadFromRun(run);
      }

      if (this.seenLogIndices.size === 0) {
        this.populateFeedFromRun(run, isPlanPreview);
      }

      this.renderPlanTab(run.test_plan || {});
      this.renderPersonasTab(run.personas || {});
      this.renderTracesTab(navResults, run.trace_artifacts || []);
      this.renderHealingTab(run.triage_report, run.healing_report);
    },

    populateFeedFromRun(run, isPlanPreview = false) {
      this.clearAgentFeed();
      const goals = run?.test_plan?.goals || run?.test_plan?.scenarios || [];
      const personas = run?.personas?.personas || [];
      const navResults = run?.navigation_results || [];
      const diagnoses = run?.triage_report?.diagnoses || [];
      const suggestions = run?.healing_report?.suggestions || [];

      let idx = 0;
      const push = (agent, message, status = 'info') => {
        this.appendAgentLog({
          index: idx++,
          agent,
          message,
          status,
          timestamp: run?.created_at || new Date().toISOString()
        });
      };

      push(
        'Planner Agent',
        `Set ${goals.length} goal(s) via ${run?.test_plan?.generated_by || 'agent'}: ${goals.map((g) => g.title).join(', ')}`,
        'success'
      );
      push(
        'Persona Agent',
        `Synthesized ${personas.length} persona(s): ${personas.map((p) => p.name).join(', ')}`,
        'success'
      );
      if (!isPlanPreview) {
        navResults.forEach((nr) => {
          (nr.steps || []).forEach((st) => {
            if (st.status === 'skipped') return;
            push(
              'Navigator Agent',
              `${nr.scenario_id} Step ${st.step_number}: ${String(st.action?.action_type || 'step').toUpperCase()} ${st.action?.selector || st.action?.value || ''} -> ${String(st.status).toUpperCase()} (${Number(st.duration_ms || 0).toFixed(0)}ms)`,
              st.status === 'passed' ? 'success' : 'error'
            );
          });
          if (nr.completion_reason) {
            const stType = nr.overall_status === 'passed' ? 'success' : (nr.overall_status === 'stuck' ? 'warning' : 'error');
            push('Navigator Agent', `${nr.scenario_id} -> ${String(nr.overall_status || 'done').toUpperCase()}: ${nr.completion_reason}`, stType);
          }
        });
        if (diagnoses.length) {
          push('Triage Agent', `Classified ${diagnoses.length} failure(s).`, 'warning');
        }
        if (suggestions.length) {
          push('Healing Agent', `Synthesized ${suggestions.length} resilient ARIA patch(es).`, 'success');
        }
      }
    },

    renderPlanTab(testPlan) {
      const container = document.getElementById('tab-panel-plan');
      if (!container) return;
      const goals = testPlan.goals && testPlan.goals.length ? testPlan.goals : (testPlan.scenarios || []);
      if (!goals.length) {
        container.innerHTML = '<p class="text-xs font-mono text-neutral-500">No goals generated yet.</p>';
        return;
      }

      const genBy = String(testPlan.generated_by || '');
      const isAi = genBy.startsWith('langchain');
      const engineBadge = genBy
        ? `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${isAi ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' : 'bg-amber-50 text-amber-800 border border-amber-200'}">${escapeHtml(isAi ? `Live AI (${genBy})` : `Fallback (${genBy})`)}</span>`
        : '';

      const summaryHtml = (testPlan.summary || engineBadge)
        ? `<div class="p-3.5 rounded-lg bg-[#f4f3f1] border border-border-hairline text-xs font-mono text-neutral-800 flex flex-wrap items-center justify-between gap-2"><span>${escapeHtml(testPlan.summary || '')}</span>${engineBadge}</div>`
        : '';

      const cardsHtml = goals.map((g, idx) => {
        const gid = g.goal_id || g.scenario_id || `GOAL-${idx + 1}`;
        const objective = g.objective || g.description || '';
        const criteria = g.success_criteria || g.expected_result || '';
        const hasSteps = Array.isArray(g.steps) && g.steps.length > 0;
        const stepsTable = hasSteps
          ? `<div class="overflow-x-auto"><table class="w-full text-left text-xs"><thead><tr class="text-[10px] font-mono uppercase text-neutral-400"><th class="pb-1.5 pr-3">#</th><th class="pb-1.5 pr-3">Action</th><th class="pb-1.5 pr-3">Description</th><th class="pb-1.5">Locator</th></tr></thead><tbody>${g.steps.map((st) => `<tr class="border-t border-border-hairline"><td class="py-1.5 pr-3 font-mono text-neutral-500">${escapeHtml(st.step_number)}</td><td class="py-1.5 pr-3 font-mono font-semibold text-black">${escapeHtml(st.action)}</td><td class="py-1.5 pr-3 text-neutral-700">${escapeHtml(st.description)}</td><td class="py-1.5 font-mono text-[11px] text-neutral-500">${escapeHtml(st.target_hint || '-')}</td></tr>`).join('')}</tbody></table></div>`
          : `<div class="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-border-hairline text-[11px] font-mono"><span class="text-neutral-600"><strong>Success criteria:</strong> ${escapeHtml(criteria || 'Verified via live DOM exploration')}</span><span class="px-2 py-0.5 rounded bg-blue-50 text-blue-800 border border-blue-200 text-[10px] font-semibold">AUTONOMOUS STEP LOOP</span></div>`;

        return `
          <div class="rounded-lg border border-border-hairline p-4 bg-white space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 rounded bg-black text-white text-[10px] font-mono">${escapeHtml(gid)}</span>
                <h4 class="font-syne font-bold text-sm text-black">${escapeHtml(g.title)}</h4>
              </div>
              <div class="flex items-center gap-2 text-[10px] font-mono">
                <span class="px-2 py-0.5 rounded bg-[#f4f3f1] text-neutral-700 uppercase">${escapeHtml(g.category || 'e2e')}</span>
                <span class="px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 uppercase">${escapeHtml(g.priority || 'medium')}</span>
              </div>
            </div>
            <p class="text-xs text-neutral-700">${escapeHtml(objective)}</p>
            ${stepsTable}
          </div>
        `;
      }).join('');

      container.innerHTML = summaryHtml + `<div class="space-y-4">${cardsHtml}</div>`;
    },

    renderPersonasTab(personasObj) {
      const container = document.getElementById('tab-panel-personas');
      if (!container) return;
      const list = personasObj.personas || [];
      if (!list.length) {
        container.innerHTML = '<p class="text-xs font-mono text-neutral-500">No personas generated yet.</p>';
        return;
      }

      container.innerHTML = `
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          ${list.map((p) => `
            <div class="rounded-lg border border-border-hairline p-4 bg-[#faf9f7] space-y-2.5">
              <div class="flex items-center justify-between gap-2">
                <span class="font-syne font-bold text-sm text-black">${escapeHtml(p.name)}</span>
                <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-border-hairline text-neutral-700">
                  ${escapeHtml(p.technical_proficiency || 'standard')}
                </span>
              </div>
              <p class="text-[11px] font-mono text-neutral-600">${escapeHtml(p.role || p.persona_id || '')}</p>
              <p class="text-xs text-neutral-700 leading-relaxed">${escapeHtml(p.primary_goal || p.behavior_notes || '')}</p>
              ${(p.edge_case_focus || p.assigned_scenario_ids) ? `
                <div class="pt-2 border-t border-border-hairline text-[10px] font-mono text-neutral-500">
                  Focus: ${escapeHtml((p.edge_case_focus || p.assigned_scenario_ids || []).join(', '))}
                </div>` : ''}
            </div>
          `).join('')}
        </div>
      `;
    },

    renderTracesTab(navResults, traceArtifacts) {
      const container = document.getElementById('tab-panel-traces');
      if (!container) return;
      if (!navResults.length && !traceArtifacts.length) {
        container.innerHTML = '<p class="text-xs font-mono text-neutral-500">No browser execution traces recorded for this preview.</p>';
        return;
      }

      const traceMap = {};
      traceArtifacts.forEach((tr) => {
        if (tr.trace_id) traceMap[tr.trace_id] = tr;
      });

      let frameCursor = 0;
      container.innerHTML = navResults.map((nr) => {
        const trace = nr.trace_id ? traceMap[nr.trace_id] : null;
        const stepsHtml = (nr.steps || []).map((st) => {
          const badgeCls = st.status === 'passed'
            ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
            : st.status === 'failed'
              ? 'bg-red-50 text-red-800 border-red-200'
              : 'bg-neutral-100 text-neutral-600 border-border-hairline';
          const playIdx = st.status !== 'skipped' ? frameCursor++ : -1;
          return `
            <div ${playIdx >= 0 ? `data-play-step-idx="${playIdx}" title="Click to jump video window to this step"` : ''}
              class="ws-trace-step-row flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2.5 rounded bg-[#faf9f7] border border-border-hairline text-xs font-mono ${playIdx >= 0 ? 'cursor-pointer hover:border-black transition-colors' : ''}">
              <div class="flex items-center gap-2.5 min-w-0">
                <span class="px-2 py-0.5 rounded border text-[10px] font-semibold uppercase ${badgeCls}">${escapeHtml(st.status)}</span>
                <span class="font-semibold text-black">${escapeHtml(st.action?.action_type || 'step')}</span>
                <span class="text-neutral-600 truncate">${escapeHtml(st.action?.selector || st.action?.value || st.action?.description || '')}</span>
              </div>
              <div class="flex items-center gap-3 text-[11px] text-neutral-500 whitespace-nowrap">
                ${st.used_fallback_selector ? `<span class="text-emerald-700">fallback: ${escapeHtml(st.used_fallback_selector)}</span>` : ''}
                ${playIdx >= 0 ? '<span class="text-[10px] text-black font-semibold"><i class="fa-solid fa-play mr-1 text-[8px]"></i>View</span>' : ''}
                <span>${Number(st.duration_ms || 0).toFixed(1)} ms</span>
              </div>
            </div>
          `;
        }).join('');

        const goalStatus = String(nr.overall_status || 'passed').toUpperCase();
        const goalCls = nr.overall_status === 'passed'
          ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
          : nr.overall_status === 'stuck'
            ? 'bg-amber-50 text-amber-900 border-amber-300'
            : 'bg-red-50 text-red-800 border-red-200';

        return `
          <div class="rounded-lg border border-border-hairline p-4 bg-white space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <div class="flex items-center gap-2">
                <span class="px-2 py-0.5 rounded bg-black text-white text-[10px]">${escapeHtml(nr.scenario_id)}</span>
                <span class="font-syne font-bold text-sm text-black">${escapeHtml(nr.scenario_title)}</span>
              </div>
              <div class="flex items-center gap-2 text-[11px]">
                <span class="px-2 py-0.5 rounded border font-bold ${goalCls}">${escapeHtml(goalStatus)}</span>
                <span class="px-2 py-0.5 rounded bg-[#f4f3f1] text-neutral-700">${escapeHtml(nr.execution_mode || 'playwright')}</span>
                ${trace ? `<span class="text-neutral-500">Trace: ${escapeHtml(trace.trace_id)}</span>` : ''}
              </div>
            </div>
            ${nr.completion_reason ? `<div class="text-[11px] font-mono text-neutral-600">${escapeHtml(nr.completion_reason)}</div>` : ''}
            <div class="space-y-2">${stepsHtml}</div>
          </div>
        `;
      }).join('');

      container.querySelectorAll('.ws-trace-step-row[data-play-step-idx]').forEach((row) => {
        row.addEventListener('click', () => {
          const idx = parseInt(row.getAttribute('data-play-step-idx'), 10);
          if (!Number.isNaN(idx)) {
            this.player.pause();
            this.player.scenarioFilter = 'ALL';
            this.player.applyFilter(false);
            this.player.goTo(idx);
            document.getElementById('ws-video-window')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        });
      });
    },

    renderHealingTab(triageReport, healingReport) {
      const output = document.getElementById('ws-triage-healing-output');
      if (!output) return;

      const diagnoses = triageReport?.diagnoses || [];
      const suggestions = healingReport?.suggestions || [];

      if (!diagnoses.length && !suggestions.length) {
        output.innerHTML = `
          <div class="p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-xs font-mono text-emerald-900">
            All executed browser checks passed cleanly — zero failures or brittle selectors needed self-healing.
            Tip: Enter a legacy CSS selector in <strong>Simulate Broken Selector</strong> above or test the standalone tool below!
          </div>
        `;
        return;
      }

      const diagHtml = diagnoses.map((d) => {
        const badgeTxt = d.needs_healing
          ? 'ELIGIBLE FOR SELF-HEALING'
          : (d.category === 'NAVIGATION_MISSING' || d.category === 'OVERLAY_BLOCKING')
            ? String(d.category).replace('_', ' ')
            : 'HUMAN REVIEW';
        const badgeCls = d.needs_healing
          ? 'bg-emerald-100 text-emerald-900'
          : (d.category === 'NAVIGATION_MISSING' || d.category === 'OVERLAY_BLOCKING')
            ? 'bg-amber-100 text-amber-900'
            : 'bg-red-100 text-red-900';
        return `
        <div class="rounded-lg border border-border-hairline p-4 bg-[#faf9f7] space-y-2 text-xs font-mono">
          <div class="flex flex-wrap items-center justify-between gap-2">
            <span class="font-bold text-black">${escapeHtml(d.failure_id)} • ${escapeHtml(d.category)}</span>
            <span class="px-2 py-0.5 rounded ${badgeCls} text-[10px] font-semibold">
              ${escapeHtml(badgeTxt)}
            </span>
          </div>
          <p class="text-neutral-700 font-sans">${escapeHtml(d.root_cause_summary)}</p>
          <div class="text-[11px] text-neutral-500">Verdict: ${escapeHtml(d.verdict)} (Confidence: ${Math.round((d.confidence || 0) * 100)}%)</div>
        </div>
      `;
      }).join('');

      const healHtml = suggestions.map((s, idx) => this.buildHealingCardHtml(s, `ws-heal-${idx}`)).join('');
      output.innerHTML = diagHtml + healHtml;
      this.bindHealingSnippetTabs(output);
    },

    buildHealingCardHtml(s, prefix) {
      const fw = s.playwright || {};
      return `
        <div class="rounded-lg border border-emerald-200 bg-white overflow-hidden text-xs font-mono" data-heal-card="${prefix}">
          <div class="px-4 py-2.5 bg-emerald-50 border-b border-emerald-200 flex flex-wrap items-center justify-between gap-2">
            <span class="font-semibold text-emerald-950">Self-Healed Patch • ${escapeHtml(s.failed_selector)}</span>
            <div class="inline-flex rounded border border-emerald-200 bg-white p-0.5 text-[10px]">
              ${['playwright', 'cypress', 'selenium'].map((name, i) => `
                <button type="button" data-fw="${name}" data-prefix="${prefix}"
                  class="ws-heal-fw-tab px-2 py-0.5 rounded ${i === 0 ? 'bg-black text-white' : 'text-neutral-600'}">${name}</button>
              `).join('')}
            </div>
          </div>
          <div class="p-4 space-y-2.5">
            <div class="text-neutral-600 font-sans text-xs">${escapeHtml(s.rationale)}</div>
            <div class="p-2.5 rounded bg-red-50 border border-red-200 text-red-900">
              - ${escapeHtml(s.failed_selector)}
            </div>
            <div class="p-2.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-950 whitespace-pre-wrap" id="${prefix}-code">
              ${escapeHtml(fw.code_snippet || s.primary_healed_locator)}
            </div>
          </div>
        </div>
      `;
    },

    bindHealingSnippetTabs(rootEl) {
      rootEl.querySelectorAll('.ws-heal-fw-tab').forEach((btn) => {
        btn.addEventListener('click', () => {
          const fwName = btn.getAttribute('data-fw');
          const prefix = btn.getAttribute('data-prefix');
          const card = rootEl.querySelector(`[data-heal-card="${prefix}"]`);
          if (!card) return;

          card.querySelectorAll('.ws-heal-fw-tab').forEach((b) => {
            b.className = b === btn
              ? 'ws-heal-fw-tab px-2 py-0.5 rounded bg-black text-white'
              : 'ws-heal-fw-tab px-2 py-0.5 rounded text-neutral-600';
          });

          const suggestions = this.state.currentRun?.healing_report?.suggestions || [];
          const idx = parseInt((prefix || '').replace('ws-heal-', ''), 10);
          const item = Number.isNaN(idx) ? this.state.lastQuickHeal : suggestions[idx];
          const codeEl = document.getElementById(`${prefix}-code`);
          if (item && item[fwName] && codeEl) {
            codeEl.textContent = item[fwName].code_snippet || item[fwName].healed_selector;
          }
        });
      });
    },

    async quickHealLocator(event) {
      event.preventDefault();
      const selector = document.getElementById('ws-qh-selector')?.value.trim();
      const actionDesc = document.getElementById('ws-qh-action')?.value.trim();
      const targetUrl = document.getElementById('ws-target-url')?.value.trim() || '';
      const resultBox = document.getElementById('ws-qh-result');
      if (!selector || !resultBox) return;

      resultBox.classList.remove('hidden');
      resultBox.innerHTML = '<div class="text-xs font-mono text-neutral-500">Synthesizing semantic locator...</div>';

      try {
        const modelChoice = AgentSwarm.models.parseSelection('ws-model-select', 'ws-custom-model');
        const res = await fetch(`${API_BASE_URL}/api/test-runs/heal-locator`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            failed_selector: selector,
            action_description: actionDesc || null,
            target_url: targetUrl,
            dom_excerpt: this.state.currentRun?.config_snapshot?.dom_snapshot || null,
            provider: modelChoice.provider,
            model_name: modelChoice.model_name
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(formatErrorMessage(data.detail, 'Quick heal failed.'));

        this.state.lastQuickHeal = data;
        resultBox.innerHTML = this.buildHealingCardHtml(data, 'ws-qh-card');
        this.bindHealingSnippetTabs(resultBox);
      } catch (err) {
        resultBox.innerHTML = `<div class="p-3 rounded bg-red-50 border border-red-200 text-xs font-mono text-red-900">${escapeHtml(err.message)}</div>`;
      }
    },

    async loadHistory() {
      const container = document.getElementById('ws-history-list');
      if (!container) return;

      try {
        const res = await fetch(`${API_BASE_URL}/api/test-runs`);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const runs = await res.json();
        this.state.runsHistory = runs;

        if (!runs.length) {
          container.innerHTML = '<div class="p-4 text-neutral-500">No runs yet. Paste a website URL and launch AgentSwarm!</div>';
          return;
        }

        container.innerHTML = runs.map((r) => {
          const statusCls = r.status === 'healed'
            ? 'text-emerald-800 bg-emerald-50'
            : r.status === 'failed'
              ? 'text-red-800 bg-red-50'
              : 'text-neutral-800 bg-[#f4f3f1]';
          return `
            <div class="p-3.5 hover:bg-[#faf9f7] transition-colors space-y-2">
              <div class="flex items-center justify-between gap-2">
                <button type="button" data-load-run="${escapeHtml(r.id)}" class="ws-load-run font-semibold text-black hover:underline text-left truncate">
                  ${escapeHtml(r.id)}
                </button>
                <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase ${statusCls}">${escapeHtml(r.status)}</span>
              </div>
              <div class="text-[11px] text-neutral-600 truncate" title="${escapeHtml(r.target_url)}">${escapeHtml(r.target_url)}</div>
              ${r.model_name ? `<div class="text-[10px] text-neutral-500 truncate">Model: ${escapeHtml(r.model_name)}</div>` : ''}
              <div class="flex items-center justify-between text-[10px] text-neutral-400">
                <span>${r.scenarios_count} scenarios • ${r.healed_count} healed</span>
                <div class="flex items-center gap-2">
                  <button type="button" data-rerun-id="${escapeHtml(r.id)}" class="ws-rerun-btn text-neutral-600 hover:text-black" title="Re-run">
                    <i class="fa-solid fa-play"></i>
                  </button>
                  <button type="button" data-delete-id="${escapeHtml(r.id)}" class="ws-delete-btn text-neutral-500 hover:text-red-600" title="Delete">
                    <i class="fa-solid fa-trash"></i>
                  </button>
                </div>
              </div>
            </div>
          `;
        }).join('');

        container.querySelectorAll('.ws-load-run').forEach((btn) => {
          btn.addEventListener('click', () => this.loadRunById(btn.getAttribute('data-load-run')));
        });
        container.querySelectorAll('.ws-rerun-btn').forEach((btn) => {
          btn.addEventListener('click', () => this.rerunById(btn.getAttribute('data-rerun-id')));
        });
        container.querySelectorAll('.ws-delete-btn').forEach((btn) => {
          btn.addEventListener('click', () => this.deleteById(btn.getAttribute('data-delete-id')));
        });
      } catch (_) {
        container.innerHTML = '<div class="p-4 text-neutral-500">Unable to load run history from backend.</div>';
      }
    },

    async loadRunById(runId) {
      if (!runId) return;
      try {
        const res = await fetch(`${API_BASE_URL}/api/test-runs/${encodeURIComponent(runId)}`);
        if (!res.ok) return;
        const data = await res.json();
        this.state.currentRun = data;
        this.seenLogIndices = new Set();
        const input = document.getElementById('ws-target-url');
        if (input && data.target_url) input.value = data.target_url;
        this.renderRunResult(data);
        this.stopPipelineAnimation(data);
      } catch (_) {}
    },

    async rerunById(runId) {
      if (!runId) return;
      this.showBanner(`Re-running ${runId}...`, 'info');
      try {
        const res = await fetch(`${API_BASE_URL}/api/test-runs/${encodeURIComponent(runId)}/rerun`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(formatErrorMessage(data.detail, 'Re-run failed.'));
        this.state.currentRun = data;
        this.renderRunResult(data);
        this.stopPipelineAnimation(data);
        this.showBanner(`Re-run ${data.id} finished (${data.status.toUpperCase()}).`, 'success');
        await this.loadHistory();
      } catch (err) {
        this.showBanner(err.message || 'Re-run failed.', 'error');
      }
    },

    async deleteById(runId) {
      if (!runId) return;
      try {
        await fetch(`${API_BASE_URL}/api/test-runs/${encodeURIComponent(runId)}`, { method: 'DELETE' });
        await this.loadHistory();
      } catch (_) {}
    }
  }
};

// Framework code diffs (Playwright / Cypress / Selenium)
const frameworkPatches = {
  playwright: {
    filename: 'e2e/checkout-journey.spec.ts',
    comment: '// Step 3: Trigger checkout transaction',
    oldLine: "const checkoutBtn = page.locator('#checkout_cta_v1');",
    errorComment: '// Error: element not found after timeout 5000ms',
    newLine: "const checkoutBtn = page.getByRole('button', { name: /proceed to review/i });",
    healedComment: '// Healed: Semantic accessible role query, resilient to DOM refactoring',
    action1: 'await checkoutBtn.click();',
    action2: 'await expect(page).toHaveURL(/.*checkout\\/summary/);',
    execNote: 'Re-executed in isolated Chromium instance (0.42s)'
  },
  cypress: {
    filename: 'cypress/e2e/checkout-journey.cy.ts',
    comment: '// Step 3: Trigger checkout transaction',
    oldLine: "cy.get('#checkout_cta_v1').click();",
    errorComment: '// AssertionError: Timed out retrying after 4000ms',
    newLine: "cy.findByRole('button', { name: /proceed to review/i }).click();",
    healedComment: '// Healed: Testing Library ARIA role selector',
    action1: "cy.location('pathname').should('include', '/checkout/summary');",
    action2: "cy.contains('Order Summary').should('be.visible');",
    execNote: 'Re-executed in isolated Cypress Electron runner (0.49s)'
  },
  selenium: {
    filename: 'tests/e2e/test_checkout_journey.py',
    comment: '# Step 3: Trigger checkout transaction',
    oldLine: 'checkout_btn = driver.find_element(By.ID, "checkout_cta_v1")',
    errorComment: '# NoSuchElementException: Unable to locate element: #checkout_cta_v1',
    newLine: 'checkout_btn = driver.find_element(By.XPATH, "//button[contains(., \'proceed to review\')]")',
    healedComment: '# Healed: Semantic accessible button matcher',
    action1: 'checkout_btn.click()',
    action2: 'WebDriverWait(driver, 5).until(EC.url_contains("/checkout/summary"))',
    execNote: 'Re-executed in isolated Selenium 4 Grid node (0.54s)'
  }
};

// Initialize Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  // 1. Target URL Form Submission (Landing page index.html)
  const targetForm = document.getElementById('swarm-target-form');
  const targetInput = document.getElementById('target-url-input');
  const openWorkspaceBtn = document.getElementById('btn-open-workspace');
  if (targetInput && openWorkspaceBtn) {
    targetInput.addEventListener('input', () => {
      const val = targetInput.value.trim();
      openWorkspaceBtn.href = val ? `workspace.html?url=${encodeURIComponent(val)}` : 'workspace.html';
    });
  }
  if (targetForm) {
    targetForm.addEventListener('submit', (e) => {
      e.preventDefault();
      AgentSwarm.swarm.deploySwarm(targetInput ? targetInput.value : '');
    });
  }

  // 2. Agent Node Card Clicks
  document.querySelectorAll('.agent-node').forEach((card) => {
    card.addEventListener('click', () => {
      const idx = parseInt(card.getAttribute('data-agent-index') || '1', 10);
      AgentSwarm.swarm.selectAgent(idx);
    });
  });

  // 3. Framework Tabs (Playwright / Cypress / Selenium)
  document.querySelectorAll('.framework-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const fw = tab.getAttribute('data-fw') || 'playwright';
      document.querySelectorAll('.framework-tab').forEach((t) => {
        if (t.getAttribute('data-fw') === fw) {
          t.className = 'framework-tab px-2 py-0.5 rounded bg-black text-white font-medium transition-colors';
        } else {
          t.className = 'framework-tab px-2 py-0.5 rounded text-neutral-600 hover:text-black transition-colors';
        }
      });
      const patch = frameworkPatches[fw];
      if (!patch) return;
      const setTxt = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      };
      setTxt('patch-filename', patch.filename);
      setTxt('patch-comment', patch.comment);
      setTxt('patch-old-line', patch.oldLine);
      setTxt('patch-error-comment', patch.errorComment);
      setTxt('patch-new-line', patch.newLine);
      setTxt('patch-healed-comment', patch.healedComment);
      setTxt('patch-action-line-1', patch.action1);
      setTxt('patch-action-line-2', patch.action2);
      setTxt('patch-exec-note', patch.execNote);
    });
  });

  // 4. Modal Event Listeners
  const signInBtn = document.getElementById('btn-signin');
  if (signInBtn) {
    signInBtn.addEventListener('click', () => AgentSwarm.authModal.open('login'));
  }

  const closeAuthBtn = document.getElementById('btn-close-auth-modal');
  if (closeAuthBtn) {
    closeAuthBtn.addEventListener('click', () => AgentSwarm.authModal.close());
  }

  const authBackdrop = document.getElementById('auth-modal-backdrop');
  if (authBackdrop) {
    authBackdrop.addEventListener('click', (e) => {
      if (e.target === authBackdrop) AgentSwarm.authModal.close();
    });
  }

  const tabLogin = document.getElementById('tab-login');
  if (tabLogin) {
    tabLogin.addEventListener('click', () => AgentSwarm.authModal.switchTab('login'));
  }

  const tabSignup = document.getElementById('tab-signup');
  if (tabSignup) {
    tabSignup.addEventListener('click', () => AgentSwarm.authModal.switchTab('signup'));
  }

  const authForm = document.getElementById('auth-form');
  if (authForm) {
    authForm.addEventListener('submit', (e) => AgentSwarm.authModal.handleSubmit(e));
  }

  const demoCredsBtn = document.getElementById('btn-use-demo-creds');
  if (demoCredsBtn) {
    demoCredsBtn.addEventListener('click', () => {
      AgentSwarm.authModal.switchTab('login');
      const emailInput = document.getElementById('auth-email');
      const passInput = document.getElementById('auth-password');
      if (emailInput) emailInput.value = 'demo@agentswarm.dev';
      if (passInput) passInput.value = 'password123';
      AgentSwarm.authModal.showAlert('Demo credentials filled: click "Sign In to Swarm" to test.', 'success');
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') AgentSwarm.authModal.close();
  });

  // Check existing session status on page load
  AgentSwarm.auth.checkAuthStatus();

  // Populate AI model dropdowns on landing page or workspace
  AgentSwarm.models.loadModels();

  // Initialize dedicated Live Workspace controller if on workspace.html
  AgentSwarm.workspace.init();
});

// Expose global namespace for backend and client access
window.AgentSwarm = AgentSwarm;
