/**
 * Dynamic AutoTuner provider for Pi.
 *
 * Install this file under ~/.pi/agent/extensions/autotuner.ts, enable
 * AutoTuner's External control API, then /reload Pi. The extension discovers
 * the current GGUF catalogue and sends each request through AutoTuner's
 * authenticated loopback gateway; selecting a different model therefore
 * performs one serialized stop/configure/start/health-check transition.
 *
 * Credential discovery order (first hit wins):
 *   1. AUTOTUNER_API_URL / AUTOTUNER_API_KEY
 *   2. AUTOTUNER_CONTROL_API_PORT / AUTOTUNER_CONTROL_API_KEY
 *   3. <AUTOTUNER_DATA_DIR|~/.autotuner>/control_api.json (AutoTuner >= 5.3.9,
 *      a tiny sidecar rewritten whenever the gateway starts, stops, or the
 *      token is regenerated)
 *   4. <AUTOTUNER_DATA_DIR|~/.autotuner>/autotuner_settings.json (older
 *      AutoTuner versions; scanned with a bounded regex because the file can
 *      hold tens of megabytes of benchmark evidence and must never be
 *      JSON.parse'd on Pi's startup path)
 *   5. http://127.0.0.1:1233 without a key (provider stays empty)
 */

import type {
  ExtensionAPI,
  ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";
import { open, readFile, stat } from "node:fs/promises";
import { request as httpRequest } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";

interface Discovery {
  port?: number;
  token?: string;
  enabled?: boolean;
}

interface AutoTunerModel {
  id: string;
  name?: string;
  context_window?: number;
  max_tokens?: number;
  input?: unknown;
}

interface ModelList {
  data?: unknown;
}

const PROVIDER_ID = "autotuner";
const DEFAULT_PORT = 1233;
const MAX_CATALOGUE_BYTES = 4 * 1024 * 1024;
const MAX_SIDECAR_BYTES = 64 * 1024;
// Only the head of the settings file is scanned; AutoTuner writes the small
// control_api_* keys in the same top-level object as its benchmark evidence,
// so a bounded window plus a targeted regex is enough and stays fast.
const MAX_SETTINGS_SCAN_BYTES = 8 * 1024 * 1024;
const COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function dataDir(): string {
  return process.env.AUTOTUNER_DATA_DIR?.trim() || join(homedir(), ".autotuner");
}

function validPort(value: unknown): number | undefined {
  const port = Number(value);
  return Number.isInteger(port) && port >= 1024 && port <= 65535
    ? port
    : undefined;
}

function validToken(value: unknown): string | undefined {
  return typeof value === "string" && value.trim().length >= 16
    ? value.trim()
    : undefined;
}

async function readSidecar(): Promise<Discovery> {
  try {
    const path = join(dataDir(), "control_api.json");
    const info = await stat(path);
    if (!info.isFile() || info.size > MAX_SIDECAR_BYTES) return {};
    const value: unknown = JSON.parse(await readFile(path, "utf8"));
    if (!value || typeof value !== "object") return {};
    const doc = value as Record<string, unknown>;
    return {
      enabled: doc.enabled === true,
      port: validPort(doc.port),
      token: doc.enabled === true ? validToken(doc.token) : undefined,
    };
  } catch {
    return {};
  }
}

async function scanLegacySettings(): Promise<Discovery> {
  // Fallback for AutoTuner < 5.3.9. The settings file may be very large, so
  // read a bounded window and look only for the two scalar keys.
  const path = join(dataDir(), "autotuner_settings.json");
  let handle: Awaited<ReturnType<typeof open>> | undefined;
  try {
    const info = await stat(path);
    if (!info.isFile()) return {};
    handle = await open(path, "r");
    const size = Math.min(info.size, MAX_SETTINGS_SCAN_BYTES);
    const buffer = Buffer.alloc(size);
    const { bytesRead } = await handle.read(buffer, 0, size, 0);
    const text = buffer.subarray(0, bytesRead).toString("utf8");
    const port = /"control_api_port"\s*:\s*(\d{4,5})/.exec(text);
    const token = /"control_api_token"\s*:\s*"([^"\\]{16,512})"/.exec(text);
    return {
      port: validPort(port?.[1]),
      token: validToken(token?.[1]),
    };
  } catch {
    return {};
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

function normalizeRoot(value: string): string {
  const parsed = new URL(value);
  if (parsed.protocol !== "http:") {
    throw new Error(
      "AUTOTUNER_API_URL must use http on the local loopback interface",
    );
  }
  const host = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, "");
  if (host !== "127.0.0.1" && host !== "localhost" && host !== "::1") {
    throw new Error("AUTOTUNER_API_URL must point to a loopback address");
  }
  parsed.username = "";
  parsed.password = "";
  parsed.search = "";
  parsed.hash = "";
  parsed.pathname = parsed.pathname.replace(/\/+$/, "").replace(/\/v1$/, "");
  return parsed.toString().replace(/\/$/, "");
}

function mapModel(raw: unknown): ProviderModelConfig | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const model = raw as AutoTunerModel;
  if (typeof model.id !== "string" || !model.id) return undefined;
  const contextWindow =
    typeof model.context_window === "number" &&
    Number.isFinite(model.context_window)
      ? Math.max(1024, Math.floor(model.context_window))
      : 8192;
  const maxTokens =
    typeof model.max_tokens === "number" && Number.isFinite(model.max_tokens)
      ? Math.max(256, Math.min(contextWindow, Math.floor(model.max_tokens)))
      : Math.max(256, Math.min(16384, Math.floor(contextWindow / 2)));
  const advertisedInput = Array.isArray(model.input)
    ? model.input.filter(
        (value): value is "text" | "image" =>
          value === "text" || value === "image",
      )
    : [];
  const input: ("text" | "image")[] =
    advertisedInput.length > 0 ? advertisedInput : ["text"];

  return {
    id: model.id,
    name:
      typeof model.name === "string" && model.name ? model.name : model.id,
    // AutoTuner owns reasoning/template launch settings. Advertising Pi-level
    // reasoning would make generic OpenAI clients send reasoning_effort, which
    // many llama.cpp builds intentionally do not implement.
    reasoning: false,
    input,
    cost: COST,
    contextWindow,
    maxTokens,
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsUsageInStreaming: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    },
  };
}

function getJson(
  url: string,
  token: string,
  signal: AbortSignal,
): Promise<ModelList> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error?: Error, value?: ModelList) => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else resolve(value ?? {});
    };
    const request = httpRequest(
      url,
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "application/json",
          Connection: "close",
        },
      },
      (response) => {
        const chunks: Buffer[] = [];
        let size = 0;
        response.on("data", (chunk: Buffer) => {
          size += chunk.length;
          if (size > MAX_CATALOGUE_BYTES) {
            request.destroy(new Error("AutoTuner model catalogue is too large"));
            return;
          }
          chunks.push(chunk);
        });
        response.on("end", () => {
          const status = response.statusCode ?? 0;
          if (status < 200 || status >= 300) {
            finish(
              new Error(
                `AutoTuner model discovery failed with HTTP ${status}`,
              ),
            );
            return;
          }
          try {
            finish(undefined, JSON.parse(Buffer.concat(chunks).toString("utf8")));
          } catch {
            finish(new Error("AutoTuner returned invalid model-catalogue JSON"));
          }
        });
        response.on("aborted", () =>
          finish(new Error("AutoTuner model discovery response was interrupted")),
        );
        response.on("error", (error) => finish(error));
      },
    );
    const onAbort = () => request.destroy(new Error("AutoTuner discovery aborted"));
    signal.addEventListener("abort", onAbort, { once: true });
    request.setTimeout(3000, () =>
      request.destroy(new Error("AutoTuner model discovery timed out")),
    );
    request.on("error", (error) => finish(error));
    if (signal.aborted) onAbort();
    else request.end();
  });
}

async function fetchModels(
  root: string,
  token: string,
  signal: AbortSignal,
): Promise<ProviderModelConfig[]> {
  const payload = await getJson(`${root}/v1/models`, token, signal);
  if (!Array.isArray(payload.data) || payload.data.length > 10_000) {
    throw new Error("AutoTuner returned an invalid model catalogue");
  }
  return payload.data
    .map(mapModel)
    .filter((model): model is ProviderModelConfig => model !== undefined);
}

export default async function (pi: ExtensionAPI) {
  const envPort = validPort(process.env.AUTOTUNER_CONTROL_API_PORT?.trim());
  const envToken =
    validToken(process.env.AUTOTUNER_API_KEY) ??
    validToken(process.env.AUTOTUNER_CONTROL_API_KEY);
  let discovered: Discovery = {};
  if (envPort === undefined || envToken === undefined) {
    discovered = await readSidecar();
    if (discovered.token === undefined && discovered.enabled !== false) {
      const legacy = await scanLegacySettings();
      discovered = {
        enabled: discovered.enabled,
        port: discovered.port ?? legacy.port,
        token: legacy.token,
      };
    }
  }
  const port = envPort ?? discovered.port ?? DEFAULT_PORT;
  // Explicit URL has highest precedence; otherwise the shared control-port
  // environment override wins over the persisted setting, exactly as in Python.
  const root = normalizeRoot(
    process.env.AUTOTUNER_API_URL?.trim() || `http://127.0.0.1:${port}`,
  );
  const token = envToken ?? discovered.token ?? "";

  let models: ProviderModelConfig[] = [];
  if (token.length >= 16) {
    try {
      models = await fetchModels(root, token, new AbortController().signal);
    } catch (error) {
      // Do not prevent Pi startup when AutoTuner is closed or still scanning.
      // /model's refresh action invokes refreshModels below.
      console.warn(
        `[AutoTuner] Dynamic model discovery is not ready: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  } else if (discovered.enabled === false) {
    console.warn(
      "[AutoTuner] The External control API is disabled. Enable it in AutoTuner Settings, then /reload Pi.",
    );
  } else {
    console.warn(
      "[AutoTuner] No API key found. Enable External control API in AutoTuner Settings, then /reload Pi.",
    );
  }

  pi.registerProvider(PROVIDER_ID, {
    name: "AutoTuner",
    baseUrl: `${root}/v1`,
    // A concrete value is required by Pi's model registry even for a local
    // provider. The fallback can never authenticate and is used only while the
    // empty, not-yet-configured provider is registered.
    apiKey: token.length >= 16 ? token : "autotuner-not-configured",
    authHeader: true,
    api: "openai-completions",
    models,
    async refreshModels({ signal }) {
      if (token.length < 16) return [];
      return fetchModels(root, token, signal);
    },
  });
}
