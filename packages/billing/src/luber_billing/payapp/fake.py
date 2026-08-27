"""A PayApp that never touches the network.

Every automated test in this repository runs against this. That is not a
convenience — a suite that could reach ``api.payapp.kr`` with real
credentials is a suite that can register a real recurring contract
against a real person's phone number, and no amount of care about which
tests do it makes that acceptable.

It ships in the package rather than in a test directory because the
reconciliation CLI and any future operator tooling need the same
deterministic double, and a fake that lives in one package's tests
gets copied into the next one.

It also does the *unhelpful* things a real provider does: times out,
returns success without the identifiers, refuses with an error code.
Those are the paths that matter — a payment integration that has only
ever been tested against its happy path is a payment integration that
has not been tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from luber_billing.payapp.client import PayAppError, RebillRegistration


@dataclass
class RegistrationCall:
    """One recorded ``rebillRegist``, so tests can assert what was sent.

    Chiefly so a test can prove the *price* came from the plan table and
    not from a request body — the assertion that matters most in the
    whole integration.
    """

    goodname: str
    goodprice: int
    recvphone: str
    cycle_day: int
    expire_date: str
    feedbackurl: str
    failurl: str
    returnurl: str
    var1: str
    var2: str | None = None


@dataclass
class FakePayAppClient:
    """Deterministic, inspectable, offline."""

    #: Identifiers handed out in order, so a test can predict them.
    next_rebill_no: int = 900001
    payurl_template: str = "https://payapp.kr/pay/{rebill_no}"

    #: Set to make the next registration fail the way PayApp would.
    fail_registration_with: PayAppError | None = None
    #: Set to make the next registration succeed but answer without
    #: identifiers — the shape the documentation does not describe, and
    #: the one the client must refuse rather than guess at.
    registration_returns_nothing: bool = False
    #: Set to make cancellation fail, so a test can prove the local state
    #: is left alone when the provider refuses.
    fail_cancel_with: PayAppError | None = None

    registrations: list[RegistrationCall] = field(default_factory=list)
    cancellations: list[str] = field(default_factory=list)

    async def register_recurring(
        self,
        *,
        goodname: str,
        goodprice: int,
        recvphone: str,
        cycle_day: int,
        expire_date: str,
        feedbackurl: str,
        failurl: str,
        returnurl: str,
        var1: str,
        var2: str | None = None,
    ) -> RebillRegistration:
        self.registrations.append(
            RegistrationCall(
                goodname=goodname,
                goodprice=goodprice,
                recvphone=recvphone,
                cycle_day=cycle_day,
                expire_date=expire_date,
                feedbackurl=feedbackurl,
                failurl=failurl,
                returnurl=returnurl,
                var1=var1,
                var2=var2,
            )
        )
        if self.fail_registration_with is not None:
            raise self.fail_registration_with
        if self.registration_returns_nothing:
            raise PayAppError("payapp accepted the registration but returned no rebill_no/payurl")
        rebill_no = str(self.next_rebill_no)
        self.next_rebill_no += 1
        return RebillRegistration(
            rebill_no=rebill_no, payurl=self.payurl_template.format(rebill_no=rebill_no)
        )

    async def cancel_recurring(self, *, rebill_no: str) -> None:
        self.cancellations.append(rebill_no)
        if self.fail_cancel_with is not None:
            raise self.fail_cancel_with


def feedback_payload(
    *,
    userid: str,
    linkkey: str,
    linkval: str,
    rebill_no: str,
    mul_no: str = "5551234",
    price: int = 29900,
    pay_state: int = 4,
    correlation_id: str | None = None,
    goodname: str = "BOORDA Pro",
    **extra: str,
) -> dict[str, str]:
    """A notification shaped the way PayApp sends them.

    ``extra`` is open on purpose: tests use it to add fields PayApp does
    not currently document, proving the parser tolerates them rather than
    rejecting a real payment on the day PayApp ships a new one.
    """
    payload = {
        "userid": userid,
        "linkkey": linkkey,
        "linkval": linkval,
        "pay_state": str(pay_state),
        "mul_no": mul_no,
        "rebill_no": rebill_no,
        "price": str(price),
        "goodname": goodname,
        "pay_date": "2026-08-27 12:00:00",
        "pay_type": "1",
    }
    if correlation_id is not None:
        payload["var1"] = correlation_id
    payload.update(extra)
    return payload


__all__ = ["FakePayAppClient", "RegistrationCall", "feedback_payload"]
