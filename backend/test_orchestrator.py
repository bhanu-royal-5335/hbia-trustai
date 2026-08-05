import asyncio
from app.layers.l0_orchestrator.orchestrator import orchestrator

async def main():
    r = await orchestrator.process('who is current education minister of andhra pradesh')
    print("Total Latency:", r.latency_ms)
    print("Layer Traces:", r.layer_traces)

if __name__ == "__main__":
    asyncio.run(main())
