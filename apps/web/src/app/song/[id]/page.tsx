interface SongPageProps {
  params: Promise<{ id: string }>;
}

export default async function SongPage({ params }: SongPageProps) {
  const { id } = await params;
  return (
    <section className="flex flex-col gap-4">
      <h1 className="text-3xl font-bold tracking-tight">Song</h1>
      <p className="text-zinc-400">
        Song detail (waveform, playback, downloads) for <span className="font-mono">{id}</span>{" "}
        arrives with the generation result UI.
      </p>
    </section>
  );
}
