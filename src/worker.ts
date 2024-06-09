import { CollapseState, WaveFunctionCollapse } from "../algorithm/pkg/";
import SimpleKnotUrl from "./assets/SimpleKnot.png";
//import * as wasm from "../algorithm/pkg/wave_function_collapse_bg.wasm";

console.log("Hello from JS worker!");
postMessage("ready");
let collapseLoopTimeout: number | null = null;
let wfc: WaveFunctionCollapse | null = null;
let canvas: OffscreenCanvas | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let start: number | null = null;

onmessage = (event) => {
	console.log("Worker received message:");
	console.log(JSON.stringify(event.data));
	if (event.data.canvas instanceof OffscreenCanvas) {
		canvas = event.data.canvas as OffscreenCanvas;
		ctx = canvas.getContext("2d");
	} else if (event.data === "start") {
		collapse();
	} else if (event.data === "pause") {
		if (collapseLoopTimeout != null) {
			clearTimeout(collapseLoopTimeout);
		}
	} else if (event.data === "resume") {
		if (collapseLoopTimeout != null) {
			if (wfc != null && ctx != null && start != null) {
				collapseLoop(wfc, ctx, start);
			}
		}
	}
};

onerror = (event) => {
	console.error("Worker had error");
	console.error(event);
};

onmessageerror = (event) => {
	console.error("Worker had error js");
	console.error(event);
};

async function collapse() {
	console.log("Collapsing...");
	if (ctx == null) {
		throw new Error("Unable to get OffscreenCanvas context!");
	}

	const imgBytes = new Uint8Array(
		await (await fetch(SimpleKnotUrl)).arrayBuffer(),
	);
	wfc = new WaveFunctionCollapse(imgBytes, 3, 3, 100, 100, false, 18);
	wfc.draw(ctx);

	start = performance.now();
	collapseLoop(wfc, ctx, start);
}

function collapseLoop(
	wfc: WaveFunctionCollapse,
	ctx: OffscreenCanvasRenderingContext2D,
	loopStart: number,
	iter = 0,
) {
	collapseLoopTimeout = setTimeout(() => {
		const tickResult = wfc.collapse_for_ticks(10);

		if (tickResult !== CollapseState.InProgress) {
			console.log(`Exiting with tick ${tickResult}`);
			requestAnimationFrame(() => draw(ctx, wfc));
			const end = performance.now();
			console.log(iter, loopStart, end, (end - loopStart) / 1000);
			return;
		}

		requestAnimationFrame(() => draw(ctx, wfc));
		collapseLoop(wfc, ctx, iter + 1);
	}, 0);
}

function draw(
	ctx: OffscreenCanvasRenderingContext2D,
	wfc: WaveFunctionCollapse,
) {
	console.log("drawing");
	wfc.draw_changes(ctx);
	//const colorsPtr = wfc.colors;
	//const colors = new Uint8Array(wasm.memory.buffer, colorsPtr, 4 * 100 * 100);
	//for (let y = 0; y < 100; y++) {
	//for (let x = 0; x < 100; x++) {
	//const color = [
	//colors[4 * (x + 100 * y)],
	//colors[4 * (x + 100 * y) + 1],
	//colors[4 * (x + 100 * y) + 2],
	//colors[4 * (x + 100 * y) + 3],
	//];
	//ctx.fillStyle = `rgb(${255}, ${color[1]}, ${color[2]})`;
	//ctx.fillRect(x * 10, y * 10, 10, 10);
	//}
	//}
}
