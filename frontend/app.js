/**
 * AgentSwarm Core Client Architecture
 * Connects frontend user flows with FastAPI & PostgreSQL backend.
 */

// Dynamic Backend API Base URL detection
const API_BASE_URL = (
  window.location.origin && window.location.origin.startsWith('http') && !window.location.origin.includes(':5500')
) ? window.location.origin : 'http://localhost:8000';

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
              href="#orchestrator" id="btn-header-workspace">
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
            href="#cta" id="btn-header-workspace">
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
  // 5. Swarm Execution & Headless Orchestrator API
  // -----------------------------------------------------------------------
  swarm: {
    state: {
      targetUrl: 'https://staging.store.nexus',
      activeAgentIndex: 1, // Default: Workflow Agent (02)
      framework: 'playwright',
      isRunning: false,
      intervalId: null
    },

    async deploySwarm(rawUrl) {
      let url = (rawUrl || '').trim();
      if (!url) url = 'https://staging.store.nexus';
      if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
      this.state.targetUrl = url;

      const urlInput = document.getElementById('target-url-input');
      if (urlInput) urlInput.value = url;

      const deployBtnLabel = document.getElementById('btn-deploy-swarm-label');
      const statusText = document.getElementById('cluster-status-text');

      if (deployBtnLabel) deployBtnLabel.textContent = 'Running...';
      if (statusText) statusText.textContent = 'SWARM ACTIVE';

      let step = 0;
      this.selectAgent(step);

      if (this.state.intervalId) clearInterval(this.state.intervalId);
      this.state.isRunning = true;

      this.state.intervalId = setInterval(() => {
        step++;
        if (step < 5) {
          AgentSwarm.swarm.selectAgent(step);
        } else {
          clearInterval(AgentSwarm.swarm.state.intervalId);
          AgentSwarm.swarm.state.intervalId = null;
          AgentSwarm.swarm.state.isRunning = false;
          AgentSwarm.swarm.selectAgent(1);

          if (deployBtnLabel) deployBtnLabel.textContent = 'Deploy';
          if (statusText) statusText.textContent = 'CLUSTER ONLINE';
        }
      }, 1100);
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

      const active = agents[index] || agents[1];

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
  // 1. Target URL Form Submission
  const targetForm = document.getElementById('swarm-target-form');
  if (targetForm) {
    targetForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const input = document.getElementById('target-url-input');
      AgentSwarm.swarm.deploySwarm(input ? input.value : '');
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
});

// Expose global namespace for backend and client access
window.AgentSwarm = AgentSwarm;
