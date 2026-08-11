import Link from "next/link";

export default function HomePage() {
  return (
    <section className="flex flex-col items-start gap-6 py-16">
      <h1 className="max-w-2xl text-5xl font-bold leading-tight tracking-tight">
        Turn a prompt and lyrics into a <span className="text-violet-400">finished song</span>.
      </h1>
      <p className="max-w-xl text-lg text-zinc-400">
        LUBER MUSIC AI generates original, downloadable music from your prompt, lyrics, and vocal
        style — rendered as studio-quality WAV and MP3.
      </p>
      <Link
        href="/create"
        className="rounded-lg bg-violet-600 px-6 py-3 text-base font-semibold text-white transition-colors hover:bg-violet-500"
      >
        Start creating
      </Link>
    </section>
  );
}
