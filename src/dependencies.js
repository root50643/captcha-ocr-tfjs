const path = require("path");

function requirePackage(name) {
  const attempts = [
    () => require(name),
    () => {
      if (!process.env.CODEX_NODE_MODULES) {
        throw new Error("CODEX_NODE_MODULES is not set.");
      }
      return require(path.join(process.env.CODEX_NODE_MODULES, name));
    }
  ];

  let lastError = null;
  for (const attempt of attempts) {
    try {
      return attempt();
    } catch (error) {
      lastError = error;
    }
  }

  throw new Error(
    `Cannot load package "${name}". Run npm install, or set CODEX_NODE_MODULES to a compatible node_modules path. Last error: ${lastError.message}`
  );
}

module.exports = { requirePackage };
