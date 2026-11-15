import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Progress badge above the Iterative Upscale nodes, fed by the backend's
// "rulz-iterative-upscale/update_status" events (core.update_node_status).
const STATUS = Symbol("iterative_upscale_status");
const STATUS_NODES = new Set(["RulZIterativeLatentUpscale", "RulZIterativeImageUpscale"]);

let listening = false;

function listen() {
	if (listening) return;
	listening = true;
	api.addEventListener("rulz-iterative-upscale/update_status", ({ detail }) => {
		const { node, progress, text } = detail;
		const n = app.graph.getNodeById(+(node || app.runningNodeId));
		if (!n || !STATUS_NODES.has(n.comfyClass)) return;
		n[STATUS] = { progress: text ? progress : null, text: text || null };
		app.canvas.setDirty(true);
	});
}

app.registerExtension({
	name: "RulZ.IterativeUpscale",

	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (!STATUS_NODES.has(nodeData.name)) return;
		listen();

		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx) {
			const r = onDrawForeground?.apply?.(this, arguments);
			const status = this[STATUS];
			if (!status?.text) return r;

			const { text, progress } = status;
			ctx.save();
			ctx.font = "12px sans-serif";
			const width = ctx.measureText(text).width + 12;
			const top = -LiteGraph.NODE_TITLE_HEIGHT - 20;

			ctx.fillStyle = "dodgerblue";
			ctx.beginPath();
			ctx.roundRect(0, top, width, 20, 5);
			ctx.fill();

			if (progress) {
				ctx.fillStyle = "green";
				ctx.beginPath();
				ctx.roundRect(0, top, width * progress, 20, 5);
				ctx.fill();
			}

			ctx.fillStyle = "#fff";
			ctx.fillText(text, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
			ctx.restore();
			return r;
		};
	},
});
