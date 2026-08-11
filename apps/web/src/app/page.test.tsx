import { render, screen } from "@testing-library/react";
import HomePage from "./page";

describe("HomePage", () => {
  it("renders the hero headline", () => {
    render(<HomePage />);
    expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    expect(screen.getByText(/finished song/i)).toBeInTheDocument();
  });

  it("links to the create page", () => {
    render(<HomePage />);
    const cta = screen.getByRole("link", { name: /start creating/i });
    expect(cta).toHaveAttribute("href", "/create");
  });
});
