(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("run-form");
  const endpoint = $("endpoint");
  const apiKey = $("api-key");
  const model = $("model");
  const prompt = $("prompt");
  const maxTokens = $("max-tokens");
  const stream = $("stream");
  const output = $("output");
  const outputShell = $("output-shell");
  const runButton = $("run-button");
  const stopButton = $("stop-button");
  const formError = $("form-error");
  let controller;
  let startedAt = 0;
  let firstTokenAt = 0;

  endpoint.value = localStorage.getItem("cie.endpoint") || window.location.origin;
  model.value = localStorage.getItem("cie.model") || model.value;
  apiKey.value = sessionStorage.getItem("cie.apiKey") || "";

  const baseUrl = () => endpoint.value.trim().replace(/\/$/, "");
  const elapsed = () => performance.now() - startedAt;
  const formatMs = (value) => value ? `${Math.round(value)} ms` : "—";

  function setRunning(running) {
    runButton.disabled = running;
    runButton.dataset.state = running ? "loading" : "idle";
    stopButton.disabled = !running;
    [endpoint, model, maxTokens, stream].forEach((field) => { field.disabled = running; });
  }

  function setError(message) {
    formError.textContent = message;
    formError.hidden = !message;
    outputShell.dataset.state = message ? "error" : output.textContent ? "ready" : "empty";
    if (message) runButton.dataset.state = "error";
  }

  function resetMetrics() {
    $("metric-ttft").textContent = "—";
    $("metric-elapsed").textContent = "—";
    $("metric-tokens").textContent = "—";
    $("metric-rate").textContent = "—";
  }

  function finishMetrics(usage) {
    const duration = elapsed();
    const tokens = usage?.output_tokens ?? "—";
    $("metric-ttft").textContent = formatMs(firstTokenAt && firstTokenAt - startedAt);
    $("metric-elapsed").textContent = formatMs(duration);
    $("metric-tokens").textContent = tokens === "—" ? tokens : `${tokens} tokens`;
    $("metric-rate").textContent = typeof tokens === "number" && duration > 0 ? `${(tokens / duration * 1000).toFixed(1)} tok/s` : "—";
  }

  async function errorMessage(response) {
    try {
      const body = await response.json();
      return body.error?.message || body.error?.code || `HTTP ${response.status}`;
    } catch {
      return `HTTP ${response.status}`;
    }
  }

  async function readStream(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let usage;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";
      for (const frame of frames) {
        const data = frame.split("\n").find((line) => line.startsWith("data:"))?.slice(5).trim();
        if (!data || data === "[DONE]") continue;
        const event = JSON.parse(data);
        if (event.type === "response.output_text.delta") {
          if (!firstTokenAt) firstTokenAt = performance.now();
          output.textContent += event.delta;
          outputShell.dataset.state = "streaming";
        } else if (event.type === "response.completed") {
          usage = event.response?.usage;
        }
      }
      if (done) return usage;
    }
  }

  function validate() {
    const fields = [endpoint, apiKey, model, prompt, maxTokens];
    fields.forEach((field) => field.removeAttribute("aria-invalid"));
    const invalid = fields.find((field) => !field.checkValidity());
    if (!invalid) return true;
    invalid.setAttribute("aria-invalid", "true");
    invalid.focus();
    setError(invalid.validationMessage || "Complete the required field and try again.");
    return false;
  }

  async function run(event) {
    event.preventDefault();
    setError("");
    if (!validate()) return;

    localStorage.setItem("cie.endpoint", baseUrl());
    localStorage.setItem("cie.model", model.value.trim());
    sessionStorage.setItem("cie.apiKey", apiKey.value);
    controller = new AbortController();
    startedAt = performance.now();
    firstTokenAt = 0;
    output.textContent = "";
    outputShell.dataset.state = "loading";
    $("run-state").textContent = "Submitting to the scheduler…";
    resetMetrics();
    setRunning(true);

    try {
      const response = await fetch(`${baseUrl()}/v1/responses`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${apiKey.value}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: model.value.trim(), input: prompt.value.trim(), max_output_tokens: Number(maxTokens.value), temperature: 0, stream: stream.checked, ...(stream.checked && { stream_options: { include_usage: true } }) }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await errorMessage(response));

      let usage;
      if (stream.checked) {
        $("run-state").textContent = "Streaming tokens…";
        usage = await readStream(response);
      } else {
        const result = await response.json();
        output.textContent = result.output?.[0]?.content?.[0]?.text || "";
        usage = result.usage;
      }
      if (!output.textContent) output.textContent = "The engine returned an empty response.";
      outputShell.dataset.state = "ready";
      $("run-state").textContent = "Completed.";
      finishMetrics(usage);
      runButton.dataset.state = "success";
    } catch (error) {
      if (error.name === "AbortError") {
        $("run-state").textContent = "Stopped by operator.";
        outputShell.dataset.state = output.textContent ? "ready" : "empty";
        finishMetrics();
      } else {
        $("run-state").textContent = "Request failed.";
        setError(`${error.message}. Check the endpoint, key, and deployed model.`);
      }
    } finally {
      setRunning(false);
      controller = undefined;
    }
  }

  async function checkHealth() {
    const health = $("health");
    try {
      const response = await fetch(`${baseUrl()}/healthz`, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error();
      const body = await response.json();
      health.dataset.state = "ready";
      $("health-label").textContent = `${body.status} · ${body.mode}`;
    } catch {
      health.dataset.state = "error";
      $("health-label").textContent = "endpoint offline";
    }
  }

  prompt.addEventListener("input", () => { $("prompt-count").textContent = prompt.value.length; });
  endpoint.addEventListener("change", checkHealth);
  form.addEventListener("submit", run);
  stopButton.addEventListener("click", () => controller?.abort());
  $("focus-prompt").addEventListener("click", () => prompt.focus());
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") form.requestSubmit();
  });
  prompt.dispatchEvent(new Event("input"));
  checkHealth();
})();
