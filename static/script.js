/* =====================================================================
   CARBON FOOTPRINT CALCULATOR — front-end logic
   Handles: theme toggle, hero stat count-up, validation, submission to
   Flask (/predict), result ring animation, reset / calculate-again.
   ===================================================================== */

document.addEventListener('DOMContentLoaded', () => {

  /* ---------------------------------------------------------------
     THEME TOGGLE
  --------------------------------------------------------------- */
  const themeToggle = document.getElementById('themeToggle');
  const root = document.documentElement;

  const applyTheme = (theme) => {
    root.setAttribute('data-theme', theme);
    themeToggle.innerHTML = theme === 'dark'
      ? '<i class="fa-solid fa-sun"></i>'
      : '<i class="fa-solid fa-moon"></i>';
  };

  // Default: follow system preference, no storage dependency required.
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(prefersDark ? 'dark' : 'light');

  themeToggle.addEventListener('click', () => {
    const current = root.getAttribute('data-theme');
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  /* ---------------------------------------------------------------
     HERO STAT COUNT-UP
  --------------------------------------------------------------- */
  const statEls = document.querySelectorAll('.stat-num');
  const countUp = (el) => {
    const target = parseFloat(el.dataset.count);
    const isDecimal = target % 1 !== 0;
    const duration = 1400;
    const start = performance.now();

    const step = (now) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = target * eased;
      el.textContent = isDecimal ? value.toFixed(1) : Math.round(value);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };
  statEls.forEach(countUp);

  /* ---------------------------------------------------------------
     FORM ELEMENTS
  --------------------------------------------------------------- */
  const form = document.getElementById('footprintForm');
  const submitBtn = document.getElementById('submitBtn');
  const resetBtn = document.getElementById('resetBtn');
  const resultCard = document.getElementById('resultCard');
  const closeResult = document.getElementById('closeResult');
  const calculateAgain = document.getElementById('calculateAgain');
  const toast = document.getElementById('toast');

  const showToast = (message, isError = false) => {
    toast.textContent = message;
    toast.classList.toggle('error', isError);
    toast.classList.add('show');
    toast.setAttribute('aria-hidden', 'false');
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => {
      toast.classList.remove('show');
      toast.setAttribute('aria-hidden', 'true');
    }, 3200);
  };

  /* ---------------------------------------------------------------
     VALIDATION
  --------------------------------------------------------------- */
  const clearFieldError = (field) => field.classList.remove('invalid');

  const validateSelectOrNumber = (input) => {
    const field = input.closest('.field');
    if (!field) return true;
    const valid = input.checkValidity() && input.value.trim() !== '';
    field.classList.toggle('invalid', !valid);
    return valid;
  };

  const validateCheckboxGroup = (groupEl) => {
    const checked = groupEl.querySelectorAll('input[type="checkbox"]:checked');
    const valid = checked.length > 0;
    groupEl.classList.toggle('invalid', !valid);
    const fieldParent = groupEl.closest('.field');
    if (fieldParent) fieldParent.classList.toggle('invalid', !valid);
    return valid;
  };

  // live-clear errors as the user fixes them
  form.querySelectorAll('select[required], input[required]').forEach((input) => {
    input.addEventListener('input', () => validateSelectOrNumber(input));
    input.addEventListener('change', () => validateSelectOrNumber(input));
  });
  form.querySelectorAll('.checkbox-group').forEach((group) => {
    group.addEventListener('change', () => validateCheckboxGroup(group));
  });

  const runFullValidation = () => {
    let allValid = true;
    let firstInvalid = null;

    form.querySelectorAll('select[required], input[required]').forEach((input) => {
      const ok = validateSelectOrNumber(input);
      if (!ok) {
        allValid = false;
        if (!firstInvalid) firstInvalid = input;
      }
    });

    ['cookingGroup', 'recyclingGroup'].forEach((id) => {
      const group = document.getElementById(id);
      const ok = validateCheckboxGroup(group);
      if (!ok) {
        allValid = false;
        if (!firstInvalid) firstInvalid = group;
      }
    });

    return { allValid, firstInvalid };
  };

  /* ---------------------------------------------------------------
     RESULT RENDERING
  --------------------------------------------------------------- */
  const RING_CIRCUMFERENCE = 2 * Math.PI * 92; // r=92

  const categoryMeta = {
    LOW:    { class: 'low',    icon: 'fa-leaf',            fraction: 0.28 },
    MEDIUM: { class: 'medium', icon: 'fa-cloud-sun',       fraction: 0.62 },
    HIGH:   { class: 'high',   icon: 'fa-triangle-exclamation', fraction: 0.92 },
  };

  const renderResult = (category, score, suggestions) => {
    const key = (category || 'MEDIUM').toUpperCase();
    const meta = categoryMeta[key] || categoryMeta.MEDIUM;

    const ring = document.getElementById('ringProgress');
    const ringIcon = document.getElementById('ringIcon');
    const ringCategory = document.getElementById('ringCategory');
    const resultCategoryText = document.getElementById('resultCategoryText');
    const suggestionsList = document.getElementById('suggestionsList');
    const predictedScore = document.getElementById("predictedScore");

    ring.setAttribute('stroke-dasharray', RING_CIRCUMFERENCE);
    ring.classList.remove('low', 'medium', 'high');
    ring.style.strokeDashoffset = RING_CIRCUMFERENCE;

    ringIcon.innerHTML = `<i class="fa-solid ${meta.icon}"></i>`;
    ringCategory.textContent = key;
    resultCategoryText.textContent = key;
    predictedScore.textContent = score;

    if (Array.isArray(suggestions) && suggestions.length) {
      suggestionsList.innerHTML = suggestions
        .map((tip) => `<li><i class="fa-solid fa-circle-check"></i> ${tip}</li>`)
        .join('');
    }

    resultCard.hidden = false;
    resultCard.classList.add('reveal');

    // animate the ring after layout settles
    requestAnimationFrame(() => {
      ring.classList.add(meta.class);
      const offset = RING_CIRCUMFERENCE * (1 - meta.fraction);
      ring.style.strokeDashoffset = offset;
    });

    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  /* ---------------------------------------------------------------
     SUBMIT — posts to Flask /predict, expects JSON:
     { category: "LOW" | "MEDIUM" | "HIGH", suggestions: [ ... ] }
     Falls back to a local placeholder if the endpoint is unavailable,
     so the page remains demoable before the backend is wired up.
  --------------------------------------------------------------- */
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const { allValid, firstInvalid } = runFullValidation();
    if (!allValid) {
      showToast('Please fill in all required fields correctly.', true);
      if (firstInvalid) {
        firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      return;
    }

    submitBtn.classList.add('loading');
    submitBtn.disabled = true;

    const formData = new FormData(form);

    try {
      const response = await fetch(form.action, {
        method: 'POST',
        body: formData,
        headers: { 'Accept': 'application/json' },
      });

      if (!response.ok) throw new Error('Server responded with an error.');

      const data = await response.json();
      renderResult(
    data.category,
    data.estimated_co2_per_month,
    data.suggestions
);

    } catch (err) {
      // Backend not reachable yet — demo fallback so the UI stays testable.
      console.warn('Prediction endpoint unavailable, showing placeholder result.', err);
      renderResult('MEDIUM', [
        'Use public transport or carpool when possible.',
        'Reduce electricity usage during peak hours.',
        'Walk or cycle for short distances.',
        'Switch to a plant-forward diet a few days a week.',
      ]);
      showToast('Showing a placeholder result — connect the Flask backend for real predictions.');
    } finally {
      submitBtn.classList.remove('loading');
      submitBtn.disabled = false;
    }
  });

  /* ---------------------------------------------------------------
     RESET / CLOSE / CALCULATE AGAIN
  --------------------------------------------------------------- */
  resetBtn.addEventListener('click', () => {
    form.reset();
    form.querySelectorAll('.invalid').forEach((el) => el.classList.remove('invalid'));
    showToast('Form cleared.');
  });

  const hideResult = () => {
    resultCard.classList.remove('reveal');
    resultCard.hidden = true;
  };

  closeResult.addEventListener('click', hideResult);

  calculateAgain.addEventListener('click', () => {
    hideResult();
    document.getElementById('calculator').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});
