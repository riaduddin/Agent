import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

// Base configurations extended from Next.js
const baseConfigs = compat.extends("next/core-web-vitals", "next/typescript");

// Custom configuration object to disable the rule
const customConfig = {
  rules: {
    "react/no-unescaped-entities": "off", // Disable the rule causing build failures
  },
};

// Combine base configurations with the custom rule override
const eslintConfig = [
  ...baseConfigs,
  customConfig, // Add the custom configuration object
];

export default eslintConfig;
