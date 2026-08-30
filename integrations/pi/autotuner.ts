/**
 * Dynamic AutoTuner provider for Pi.
 *
 * Install this file under ~/.pi/agent/extensions/autotuner.ts, enable
 * AutoTuner's External control API, then /reload Pi. The extension discovers
 * the current GGUF catalogue and sends each request through AutoTuner's
 * authenticated loopback gateway; selecting a different model therefore
 * performs one serialized stop/configure/start/health-check transition.
 */

import type {
  ExtensionAPI,
  ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";
import { readFile } from "node:fs/promises";
import { request as httpRequest } from "node:http";
import { homedir } from "node:os";
import { join } from "node:path";

interface AutoTunerSettings {
  control_api_port?: unknown;
  control_api_token?: unknown;
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
const COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

async function readSettings(): Promise<AutoTunerSettings> {
  const root =
    process.env.AUTOTUNER_DATA_DIR?.trim() || join(homedir(), ".autotuner");
  try {
    const raw = await readFile(join(root, "autotuner_settings.json"), "utf8");
    if (raw.length > 2 * 1024 * 1024) return {};
    const value: unknown = JSON.parse(raw);
    return value && typeof value === "object"
      ? (value as AutoTunerSettings)
      : {};
  } catch {
    return {};
  }
}

function validPort(value: unknown): number | undefined {
  const port = Number(value);
  return Number.isInteger(port) && port >= 1024 && port <= 65535
    ? port
    : undefined;
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
  const settings = await readSettings();
  const port =
    validPort(process.env.AUTOTUNER_CONTROL_API_PORT?.trim()) ??
    validPort(settings.control_api_port) ??
    DEFAULT_PORT;
  // Explicit URL has highest precedence; otherwise the shared control-port
  // environment override wins over the persisted setting, exactly as in Python.
  const root = normalizeRoot(
    process.env.AUTOTUNER_API_URL?.trim() || `http://127.0.0.1:${port}`,
  );
  const token =
    process.env.AUTOTUNER_API_KEY?.trim() ||
    process.env.AUTOTUNER_CONTROL_API_KEY?.trim() ||
    (typeof settings.control_api_token === "string"
      ? settings.control_api_token
      : "");

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
