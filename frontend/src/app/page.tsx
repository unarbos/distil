import {
  fetchMetagraph,
  fetchCommitments,
  fetchScores,
  fetchPrice,
  fetchAllModelInfo,
  fetchH2hLatest,
  buildMinerList,
} from "@/lib/api";
import { AutoRefresh } from "@/components/auto-refresh";
import { DashboardV2 } from "@/components/v2/dashboard-v2";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const [metagraph, commitments, scores, price, h2hLatest] = await Promise.all([
    fetchMetagraph(),
    fetchCommitments(),
    fetchScores(),
    fetchPrice(),
    fetchH2hLatest(),
  ]);

  const kingUid = h2hLatest?.king_uid ?? null;
  const miners = buildMinerList(metagraph, commitments, scores, kingUid, h2hLatest);
  const modelInfoMap = await fetchAllModelInfo(miners.map((m) => m.model));
  const currentBlock = metagraph?.block ?? 0;
  const taoUsd = price?.tao_usd ?? 0;
  const minersTaoDay = price?.miners_tao_per_day ?? 0;

  return (
    <>
      <AutoRefresh intervalMs={30_000} />
      <DashboardV2
        miners={miners}
        modelInfoMap={modelInfoMap}
        currentBlock={currentBlock}
        taoUsd={taoUsd}
        minersTaoDay={minersTaoDay}
        kingUid={kingUid}
      />
    </>
  );
}
