(async () => {
	console.log("Hello from JS!");
	const canvas = document.querySelector<HTMLCanvasElement>("#canvas");
	if (canvas == null) {
		throw new Error("Unable to get canvas");
	}
	canvas.width = 1000;
	canvas.height = 1000;
	const offscreen = canvas.transferControlToOffscreen();

	const worker = new Worker(new URL("./worker", import.meta.url), {
		type: "module",
	});
	worker.onmessage = (event) => {
		console.log("Message received:");
		console.log(event);
		if (event.data === "ready") {
			worker.postMessage({ canvas: offscreen }, [offscreen]);
		}
	};
	worker.onerror = (event) => {
		console.error("Worker had error js");
		console.error(event);
	};
	worker.onmessageerror = (event) => {
		console.error("Worker had error js");
		console.error(event);
	};

	const pauseButton = document.querySelector<HTMLButtonElement>("button#pause");
	if (pauseButton == null) {
		throw new Error("Unable to get pause button");
	}
	pauseButton.addEventListener("click", () => {
		if (pauseButton.textContent === "Pause") {
			worker.postMessage("pause");
			pauseButton.textContent = "Resume";
		} else if (pauseButton.textContent === "Resume") {
			worker.postMessage("resume");
			pauseButton.textContent = "Pause";
		} else if (pauseButton.textContent === "Start") {
			worker.postMessage("start");
			pauseButton.textContent = "Pause";
		}
	});
})();
