import { redirect } from "next/navigation";

export default function HomePage() {
  // Create is the product's front door.
  redirect("/create");
}
